# Algorithm and correspondence to the manuscript

## Fixed realized trace

The workflow induces a finite directed acyclic graph of realized invocations. Agent identities may recur; callbacks for the same identity must share the same parameters. Only ancestors of the supervised terminal node receive recursive credit. Routing, tool selection and unrealized branches are outside this estimator.

Each edge records the actual sampled sender IDs, decoded text, its range inside the complete receiver prompt, and receiver token positions overlapping that range. The framework does not replace hard text with a continuous message at deployment.

## Token-selection STE — equations 2–3

For frozen sender embedding matrix E and temperature tau:

```text
pi = softmax(logits / tau)
soft = pi @ E
hard = E[sampled_ids]
sender_st = hard + soft - stop_gradient(soft)
```

The code evaluates `hard + (soft - soft.detach())`, preserving exactly the hard forward value. Backward propagation follows the expected embedding. It avoids constructing a straight-through one-hot matrix, although full-vocabulary probabilities still consume memory.

## Byte alignment and bridge — equations 4–7

Sender intervals are local to the transmitted UTF-8 message. Receiver intervals are global to the exact completed prompt. For each overlapping receiver token, clip its interval to the message range and compute:

```text
A[j,l] = byte_overlap(receiver_j, sender_l) / sum_l byte_overlap(receiver_j, sender_l)
C_sur = A @ B(sender_st)
```

Rows without overlap remain zero; prompt-only and special tokens are excluded. The sparse overlap builder uses sorted nonoverlapping spans and rejects ambiguous spans. `M` counts selected receiver message tokens, including any zero-overlap rows, matching the manuscript's slice-length normalization.

An identity bridge requires an explicitly shared frozen embedding space. Equal dimensions alone do not establish shared space. Heterogeneous pairs receive a trainable affine map; the registry reuses it for the same directed pair.

## Receiver STE and manual VJP — equations 8, 14–15

```text
alpha = mu / M
C_hat = C_real + alpha * (C_sur - stop_gradient(C_sur))
G = stop_gradient(d J_task / d C_real)
proxy = alpha * sum(G * C_sur)
```

The explicit receiver STE and manual proxy have identical local first-order gradients for fixed trace, alignment and detached G. `test_manual_vjp_matches_explicit_second_ste` verifies both sender-logit and affine-bridge gradients in double precision.

## Reverse-topological training

The engine recomputes one node at a time using recorded hard prefixes and independent incoming embedding leaves. It forms a task objective from downstream proxies, or final-answer loss at the terminal. It extracts gradients with respect to incoming leaves before adding local stabilizers, stores detached values for earlier senders, then accumulates local parameter gradients.

This distinction matters: cosine consistency and reference-policy KL may update a sender and its bridge, but their gradients must not recursively become an earlier agent's task signal. The tests include a three-node analytic example that detects this leakage and a branching trace that checks accumulation.

For batch or Monte Carlo averaging, `loss_scale` multiplies terminal loss and each local stabilizer once. Do not multiply it again at every hop. The runnable recipe uses one sampled trace per example; additional Monte Carlo traces require repeated calls and appropriately scaled accumulation.

## Final-answer objective and optional stabilizers

`final_answer_loss` takes already-shifted logits and labels. Prompt/padding labels are `-100`. It averages valid tokens per example before averaging examples. Intermediate messages are generated without gold targets.

`consistency_loss` compares supported surrogate rows with detached real embeddings; `policy_kl` compares the current sender distribution with a detached reference under the same hard prefix and temperature. The tiny recipe disables both. Large-model callbacks must supply a frozen initial policy and restore matching forward conditions if these utilities are enabled.

## Deliberate implementation limits

The core engine reduces cross-node graph retention but does not implement transformer block checkpointing, KV-cache optimization or distributed scheduling. Callback implementations own RNG replay, model caching and batching. Tiny agents have no dropout and use CPU float32; numerical gradient-equivalence tests use float64. The manuscript's bf16 large-model setting requires separate hardware validation.
