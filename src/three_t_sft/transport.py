"""The two straight-through estimators and their first-order local VJP."""
import math
import torch
from torch import nn


def token_ste(logits, hard_ids, embedding_weight, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if logits.shape[:-1] != hard_ids.shape or logits.shape[-1] != embedding_weight.shape[0]:
        raise ValueError('Inconsistent token/logit/vocabulary shapes')
    probabilities = (logits / temperature).softmax(-1)
    soft = probabilities @ embedding_weight.detach()
    hard = nn.functional.embedding(hard_ids, embedding_weight.detach())
    # This algebraic form is exactly hard in forward floating-point arithmetic.
    return hard + (soft-soft.detach())


class Bridges(nn.Module):
    """One bridge per directed agent pair; identity requires a shared space."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.pairs = {}

    def register(self, sender, receiver, sender_dim, receiver_dim, *, shared_space=False):
        pair = (sender, receiver)
        if pair in self.pairs:
            raise ValueError('Pair already registered')
        if shared_space and sender_dim != receiver_dim:
            raise ValueError('A shared embedding space must have matching dimensions')
        key = f'pair_{len(self.pairs)}'
        self.pairs[pair] = key
        self.layers[key] = nn.Identity() if shared_space else nn.Linear(sender_dim, receiver_dim)

    def forward(self, sender, receiver, embeddings):
        return self.layers[self.pairs[(sender, receiver)]](embeddings)


def aligned_surrogate(sender_st, bridge, alignment):
    mapped = bridge(sender_st)
    if mapped.shape[0] != alignment.sender_count:
        raise ValueError('Sender token count differs from alignment')
    return torch.sparse.mm(alignment.tensor(device=mapped.device, dtype=mapped.dtype), mapped)


def receiver_ste(real, surrogate, mu=1.0):
    if real.shape != surrogate.shape or real.ndim != 2:
        raise ValueError('Expected matching [receiver tokens, embedding] tensors')
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError('Upstream scale must be finite and positive')
    if real.shape[0] == 0: return real.detach()
    return real.detach() + (mu/real.shape[0]) * (surrogate-surrogate.detach())


def manual_proxy(surrogate, receiver_gradient, mu=1.0):
    if surrogate.shape != receiver_gradient.shape or surrogate.ndim != 2:
        raise ValueError('Gradient and surrogate shapes must match')
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError('Upstream scale must be finite and positive')
    if surrogate.shape[0] == 0: return surrogate.sum()*0
    return (mu/surrogate.shape[0]) * (receiver_gradient.detach()*surrogate).sum()


def consistency_loss(surrogate, real, valid_rows):
    if not valid_rows: return surrogate.sum()*0
    indices = torch.tensor(valid_rows, device=surrogate.device)
    return (1-nn.functional.cosine_similarity(surrogate[indices], real.detach()[indices], dim=-1)).mean()


def policy_kl(logits, reference_logits, mask, temperature=1.0):
    if logits.shape != reference_logits.shape or mask.shape != logits.shape[:-1]:
        raise ValueError('KL shapes must match')
    if temperature <= 0 or not math.isfinite(temperature): raise ValueError('Invalid temperature')
    if not mask.bool().any(): return logits.sum()*0
    logp = (logits/temperature).log_softmax(-1)
    logq = (reference_logits.detach()/temperature).log_softmax(-1)
    return (logp.exp()*(logp-logq)).sum(-1)[mask.bool()].mean()


def final_answer_loss(logits, labels):
    """Already-shifted next-token logits/targets, prompt and padding = -100.

    Average supervised tokens within each example, then average examples.
    This avoids overweighting examples with longer final answers.
    """
    if logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError('Expected logits [B,L,V] and labels [B,L]')
    mask = labels != -100
    counts = mask.sum(-1)
    if (counts == 0).any(): raise ValueError('Every example needs final-answer supervision')
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
    return (loss.sum(-1)/counts).mean()
