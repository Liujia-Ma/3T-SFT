# 3T-SFT — Training Through Text

Reference implementation of **Retokenization-Aware Double Straight-Through Supervised Fine-Tuning (RADST-SFT)** for text-communicating multi-agent systems.

The public project name is 3T-SFT; RADST-SFT is the method name used in the supplied manuscript. Training uses final-answer supervision without gold intermediate messages. Agents continue to exchange ordinary hard text at inference.

Implementation is organized around UTF-8 span alignment, two straight-through estimators, directed embedding bridges, and a reverse-topological first-order gradient sweep. This repository is a new manuscript-based implementation, not an archive of the original experiments.
