# Local validation record

Validation performed during reference implementation development on 2026-09-29.

## Environment and checks

- Windows, Python 3.12, PyTorch 2.14.0+cpu.
- 21 automated tests passed.
- Full and terminal-only training each ran for 30 optimization steps with seed 42.
- Three tiny GRU agents, heterogeneous tokenization and embedding widths, frozen base parameters and trainable LoRA output adapters.
- Synthetic modular addition: 60 training, 20 validation, 20 test examples; split IDs are saved by each run.
- Validation loss selects the checkpoint, including the initial model as a candidate. Test data is evaluated after selection.

## Measured gradient-flow check

First-step sums of adapter gradient norms:

| Mode | First agent | Middle agent | Final agent |
|---|---:|---:|---:|
| RADST | 3.3325e-7 | 2.8232e-4 | 0.129905 |
| Terminal-only | 0 | 0 | 0.129905 |

The initial final-agent gradient is identical; only the full training method propagates credit to upstream adapters. The tests also confirm that frozen embedding tables receive no gradients and heterogeneous bridges do receive gradients.

## Synthetic training results

| Mode | Selected validation loss | Test loss | Test exact match |
|---|---:|---:|---:|
| RADST | 2.694963 | 2.725577 | 2/20 |
| Terminal-only | 2.709514 | 2.746354 | 1/20 |

These low-accuracy, single-seed runs verify execution, logging and checkpoint selection. They do not establish a task-quality advantage, statistical significance, or reproduction of the manuscript's benchmarks. Upstream gradients are small in this fixture; scale, depth and stability require measurement in real models.

## What the tests cover

Unicode byte lengths; boundary clipping; special tokens; sparse overlap weights; empty overlap; ambiguous-offset rejection; hard-forward equality; manual-VJP equivalence for sender and bridge; identity-space constraints; final-answer masking; reference-policy detachment; reverse recursion; branch accumulation; stabilizer isolation; cycle rejection; actual retokenization; homogeneous and heterogeneous training; terminal-only isolation; split disjointness; and inference without bridge use.

## Not validated here

Pretrained LLM integration, multi-GPU scheduling, bf16 behavior, tokenizer normalization edge cases from production vocabularies, dropout replay, and the original benchmark suite. The CI workflow provides repeatable CPU checks; its inclusion alone is not a claim that a remote CI run has completed.
