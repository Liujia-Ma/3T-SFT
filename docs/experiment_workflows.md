# Reproducible experiment workflows

## Run distinct seeds without editing configuration files

```bash
python scripts/train_tiny.py --seed 43 --steps 30 --output outputs/seed43-radst
python scripts/train_tiny.py --seed 43 --steps 30 --mode terminal --output outputs/seed43-terminal
```

Use a new output directory for each experiment. Existing metrics/checkpoints are
protected by default. `--overwrite` explicitly replaces artifacts in the chosen
run directory; retain previous runs when comparing results.

Effective settings and source IDs are saved with every run. Summaries include
trainable/frozen parameter counts; per-step metrics include the global gradient
norm before clipping. Nonfinite parameter gradients fail before the optimizer
step. Checkpoint replacement is atomic to reduce partial-write risk.

## Repeated-run comparisons

`metrics.summarize_seeds` reports mean and sample standard deviation. A single
seed has no estimated sample standard deviation. `paired_gains` requires exactly
matched seed IDs and summarizes within-seed differences. These helpers do not
create additional experimental runs or imply statistical significance.

## Stochastic node replay

Capture `RNGState` at the original node forward, then wrap its recomputation using
`replayable_node(node_id, callback, state)`. Replaying restores CPU and available
CUDA RNG state inside the callback, then restores the caller's state on exit.
CPU dropout replay is tested; CUDA execution needs validation on appropriate hardware.
Model state, hard prefixes and attention masks must also match the recorded trace.

## Final-answer data and masks

`data.load_final_answers` loads JSONL with unique nonempty `id`, `question` and
`answer` strings. Extra fields such as rationales are excluded from returned rows.
It is an integration utility, not a general dataset adapter wired into the ASCII
tiny task. `transport.answer_labels` masks prompt and padding positions with -100;
it returns unshifted labels, so align them with causal logits exactly once.

The tiny tokenizer still supports its declared ASCII alphabet only. Large-model
tokenizers, benchmark evaluation and distributed training remain separate integration work.
