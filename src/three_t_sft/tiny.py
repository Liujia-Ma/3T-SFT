"""Executable CPU laboratory, not a pretrained LLM or benchmark reproduction."""
from dataclasses import dataclass
import torch
from torch import nn
from .alignment import Span, align
from .transport import token_ste, aligned_surrogate, Bridges, final_answer_loss
from .engine import Node, Edge, Computation, backward_trace


class TextTokenizer:
    """Lossless ASCII tokenizer with optional two-character vocabulary items."""
    def __init__(self, pairs=False):
        self.tokens = list('0123456789+=?; ')
        if pairs: self.tokens += ['12', '23', '45', '67', '89', '0+', '1+', '2+', '3+']
        self.lookup = {t: i for i, t in enumerate(self.tokens)}

    def encode(self, text):
        ids, spans, cursor = [], [], 0
        while cursor < len(text):
            piece = text[cursor:cursor+2]
            if piece not in self.lookup: piece = text[cursor]
            if piece not in self.lookup: raise ValueError('Fixture tokenizer accepts its declared ASCII alphabet only')
            ids.append(self.lookup[piece]); spans.append(Span(cursor, cursor+len(piece)))
            cursor += len(piece)
        return torch.tensor(ids, dtype=torch.long), tuple(spans)

    def decode_with_spans(self, ids):
        pieces, spans, cursor = [], [], 0
        for token_id in ids.tolist():
            piece = self.tokens[token_id]
            pieces.append(piece); spans.append(Span(cursor, cursor+len(piece)))
            cursor += len(piece)
        return ''.join(pieces), tuple(spans)


class TinyAgent(nn.Module):
    """Frozen embedding/GRU/output projection with a role-specific LoRA head."""
    def __init__(self, tokenizer, width=24, rank=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(len(tokenizer.tokens), width)
        self.recurrent = nn.GRU(width, width, batch_first=True)
        self.output = nn.Linear(width, len(tokenizer.tokens), bias=False)
        for module in (self.embedding, self.recurrent, self.output):
            for parameter in module.parameters(): parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(rank, width)*.02)
        self.lora_b = nn.Parameter(torch.zeros(len(tokenizer.tokens), rank))
        self.scale = 2.

    def forward(self, embeddings):
        hidden, _ = self.recurrent(embeddings.unsqueeze(0))
        hidden = hidden.squeeze(0)
        return self.output(hidden) + self.scale*(hidden@self.lora_a.T@self.lora_b.T)

    def parameters_to_train(self): return (self.lora_a, self.lora_b)

    @torch.no_grad()
    def generate(self, prompt_ids, max_tokens, temperature=1., greedy=False):
        prefix, chosen = prompt_ids.clone(), []
        for _ in range(max_tokens):
            logits = self(self.embedding(prefix))[-1]
            token = logits.argmax().reshape(1) if greedy else torch.multinomial((logits/temperature).softmax(-1), 1)
            chosen.append(token); prefix = torch.cat((prefix, token))
        return torch.cat(chosen)


@dataclass
class Record:
    prompt_ids: torch.Tensor
    emitted: torch.Tensor | None
    alignment: object | None


class TinyWorkflow(nn.Module):
    def __init__(self, *, heterogeneous=True):
        super().__init__()
        self.agents = nn.ModuleList([TinyAgent(TextTokenizer(False), 24),
                                    TinyAgent(TextTokenizer(heterogeneous), 32 if heterogeneous else 24),
                                    TinyAgent(TextTokenizer(False), 24)])
        # Homogeneous means the actual frozen embedding space is shared.
        if not heterogeneous:
            for agent in self.agents[1:]: agent.embedding = self.agents[0].embedding
        self.bridges = Bridges()
        for i in range(2):
            self.bridges.register(str(i), str(i+1), self.agents[i].embedding.embedding_dim,
                                  self.agents[i+1].embedding.embedding_dim, shared_space=not heterogeneous)

    @torch.no_grad()
    def trace(self, question, *, max_tokens=4, temperature=1., greedy=False):
        records, edges = [], []
        message, sender_spans = '', ()
        for i, agent in enumerate(self.agents):
            prefix = question+';'
            prompt = prefix+message+';'
            ids, receiver_spans = agent.tokenizer.encode(prompt)
            mapping = None
            if i:
                mapping = align(sender_spans, receiver_spans, Span(len(prefix), len(prefix)+len(message)))
                real = agent.embedding(ids[list(mapping.receiver_indices)]).detach()
                edges.append(Edge(f'e{i-1}', str(i-1), str(i), real))
            emitted = agent.generate(ids, max_tokens, temperature, greedy) if i < 2 else None
            records.append(Record(ids, emitted, mapping))
            if emitted is not None: message, sender_spans = agent.tokenizer.decode_with_spans(emitted)
        return records, edges

    def programs(self, records, answer, *, temperature=1.):
        nodes = []
        for index, agent in enumerate(self.agents):
            def recompute(incoming, i=index, model=agent):
                record = records[i]
                embeddings = model.embedding(record.prompt_ids).detach().clone()
                if i:
                    indices = torch.tensor(record.alignment.receiver_indices, dtype=torch.long)
                    embeddings = embeddings.index_copy(0, indices, incoming[f'e{i-1}'])
                if i == 2:
                    logits = model(embeddings)[-1:]
                    target, _ = model.tokenizer.encode(answer)
                    if len(target) != 1: raise ValueError('Tiny task expects a one-token final answer')
                    return Computation({}, final_loss=final_answer_loss(logits.unsqueeze(0), target.reshape(1, 1)))
                # Replay logits on recorded hard autoregressive prefixes.
                continuation = model.embedding(record.emitted[:-1]).detach()
                logits = model(torch.cat((embeddings, continuation)))[len(embeddings)-1:]
                st = token_ste(logits, record.emitted, model.embedding.weight, temperature)
                mapping = records[i+1].alignment
                surrogate = aligned_surrogate(st, lambda h: self.bridges(str(i), str(i+1), h), mapping)
                return Computation({f'e{i}': surrogate})
            nodes.append(Node(str(index), recompute))
        return nodes

    def backward_example(self, question, answer, *, mode='radst', max_tokens=4, temperature=1.):
        records, edges = self.trace(question, max_tokens=max_tokens, temperature=temperature)
        nodes = self.programs(records, answer, temperature=temperature)
        if mode == 'radst': return backward_trace(nodes, edges, '2')
        if mode != 'terminal': raise ValueError('Mode must be radst or terminal')
        incoming = {e.id: e.real.detach() for e in edges if e.receiver == '2'}
        loss = nodes[-1].recompute(incoming).final_loss
        loss.backward()
        return {'final_loss': loss.detach().item(), 'visited_nodes': ['2'], 'transport_norms': {}}

    @torch.no_grad()
    def predict(self, question, max_tokens=4):
        # Deployment deliberately never calls bridge or STE modules.
        records, _ = self.trace(question, max_tokens=max_tokens, greedy=True)
        model = self.agents[-1]
        token = model(model.embedding(records[-1].prompt_ids))[-1].argmax().reshape(1)
        return model.tokenizer.decode_with_spans(token)[0]
