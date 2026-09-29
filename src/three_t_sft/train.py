"""Small reproducible final-answer-only experiment with checkpoint selection."""
import argparse
import json
import random
from pathlib import Path
import torch
from .tiny import TinyWorkflow
from .config import validate_config, apply_overrides
from .checkpoint import atomic_save
from .metrics import parameter_counts, gradient_norm


def data(seed):
    # Source IDs are disjoint across train, validation and test.
    rows = [{'id': f'add-{a}-{b}', 'question': f'{a}+{b}=', 'answer': str((a+b)%10)}
            for a in range(10) for b in range(10)]
    random.Random(seed).shuffle(rows)
    return rows[:60], rows[60:80], rows[80:]


@torch.no_grad()
def evaluate(model, rows, max_tokens):
    losses, correct = [], 0
    for row in rows:
        records, edges = model.trace(row['question'], max_tokens=max_tokens, greedy=True)
        final = model.programs(records, row['answer'])[-1]
        incoming = {e.id: e.real for e in edges if e.receiver == '2'}
        losses.append(final.recompute(incoming).final_loss.item())
        agent = model.agents[-1]
        logits = agent(agent.embedding(records[-1].prompt_ids))[-1]
        prediction = agent.tokenizer.decode_with_spans(logits.argmax().reshape(1))[0]
        correct += prediction == row['answer']
    return {'loss': sum(losses)/len(losses), 'exact_match': correct/len(rows), 'count': len(rows)}


def run(config, output, *, overwrite=False):
    config = validate_config(config)
    output = Path(output)
    if not overwrite and any((output/name).exists() for name in ('metrics.jsonl','best.pt','summary.json')):
        raise FileExistsError('Existing experiment artifacts; choose a new output or pass --overwrite')
    torch.set_num_threads(1)
    seed = config['seed']; torch.manual_seed(seed)
    model = TinyWorkflow(heterogeneous=config['heterogeneous'])
    params = list(model.bridges.parameters())
    for i, agent in enumerate(model.agents):
        for p in agent.parameters_to_train():
            p.requires_grad_(config['mode']=='radst' or i==2)
            if p.requires_grad: params.append(p)
    if config['mode']=='terminal':
        for p in model.bridges.parameters(): p.requires_grad_(False)
        params = [p for p in params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'])
    train, validation, test = data(seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    manifest = {'train': [x['id'] for x in train], 'validation': [x['id'] for x in validation],
                'test': [x['id'] for x in test]}
    (output/'splits.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    checkpoint = output/'best.pt'
    best = evaluate(model, validation, config['max_tokens'])['loss']
    atomic_save(model.state_dict(), checkpoint)
    records = []
    with (output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        for step in range(config['steps']):
            row = train[step % len(train)]
            optimizer.zero_grad(set_to_none=True)
            report = model.backward_example(row['question'], row['answer'], mode=config['mode'],
                max_tokens=config['max_tokens'], temperature=config['temperature'])
            norms = [sum(float(p.grad.norm()) for p in a.parameters_to_train() if p.grad is not None)
                     for a in model.agents]
            global_norm = gradient_norm(params)
            torch.nn.utils.clip_grad_norm_(params, config['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            metrics = {'step': step+1, 'train_loss': report['final_loss'], 'agent_grad_norms': norms,
                       'transport_norms': report['transport_norms'], 'global_grad_norm_before_clip': global_norm}
            if (step+1) % config['eval_every'] == 0 or step+1 == config['steps']:
                metrics['validation'] = evaluate(model, validation, config['max_tokens'])
                if metrics['validation']['loss'] < best:
                    best = metrics['validation']['loss']; atomic_save(model.state_dict(), checkpoint)
            log.write(json.dumps(metrics)+'\n'); records.append(metrics)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    summary = {'kind': 'synthetic_cpu_engineering_check', 'torch': torch.__version__,
               'mode': config['mode'], 'seed': seed, 'steps': config['steps'],
               'parameters': parameter_counts(model),
               'best_validation_loss': best, 'test': evaluate(model, test, config['max_tokens']),
               'first_step_agent_grad_norms': records[0]['agent_grad_norms']}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/tiny.json')
    parser.add_argument('--output', default='outputs/tiny-radst')
    parser.add_argument('--mode', choices=['radst', 'terminal'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace artifacts in this run directory')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    config = apply_overrides(config, mode=args.mode, steps=args.steps, seed=args.seed)
    if config['steps'] < 1 or config['eval_every'] < 1 or config['max_tokens'] < 1:
        raise ValueError('Positive steps, evaluation interval and message length required')
    print(json.dumps(run(config, args.output, overwrite=args.overwrite), indent=2))


if __name__ == '__main__': main()
