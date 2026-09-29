# Integrating pretrained language models

The executable tiny workflow demonstrates the contracts. To connect a real causal language model, replace its agent/tokenizer and node recomputation code while retaining alignment, transport and engine modules.

## Hard rollout and recorded state

Use a normal hard-token rollout. Save the sampled IDs, exact transmitted string, actual stop position, complete receiving prompt and message byte range. Record role templates, decoding temperature and truncation. Tokenize the entire receiver prompt; tokenizing the message alone can give incorrect boundary tokens.

When converting tokenizer offsets, establish whether they refer to Unicode characters, raw bytes, or a normalized string. `character_offsets_to_bytes` is valid only for character offsets against the exact original text. Sender offsets must describe the actual emitted IDs. Re-encoding a decoded message may yield different IDs and is not a reliable sender alignment shortcut. Reject or explicitly audit ambiguous byte-level offsets and normalization changes.

The built-in toy tokenizer is lossless ASCII and has exact offsets. It is not advertised as a general Hugging Face tokenizer adapter.

## Recompute callback

Implement `Node.recompute(incoming_slices) -> Computation`:

1. Embed the recorded complete prompt with the receiver's frozen embedding table.
2. Replace recorded message slices with the supplied independent embedding leaves.
3. Replay the recorded hard generated prefix with teacher forcing; logits must align with the sampled message token IDs.
4. For each outgoing edge, compute token STE, directed bridge and sparse alignment, returning the receiver-shaped surrogate under that edge ID.
5. At the terminal, return final-answer-only masked loss. Shift causal labels exactly once.
6. If enabled, return local stabilization separately; never add it to recursive task proxies.

The final-answer target may be teacher-forced only at the terminal. Do not include the target in upstream generation prompts or supply intermediate target messages.

## Parameter and replay discipline

Freeze base embeddings and base model parameters, and expose the selected role-specific adapters plus affine bridges to the optimizer. Homogeneous identity bridges require shared embedding weights, not merely identical dimensions. Register each directed pair before constructing the optimizer; save the pair configuration with checkpoints.

Use the same model/adapter state throughout the recorded trace and reverse sweep; call `optimizer.step()` only after the full accumulation. Restore per-node RNG state for dropout or other stochastic modules. Otherwise recomputation may not describe the recorded forward realization. The tiny example has no dropout, so RNG replay is unnecessary there.

If one receiver token overlaps two message ranges, define and audit a composition rule before integrating multiple incoming messages. The generic engine supports multiple incoming edges, but the included prompt construction is a single-parent chain and does not implement overlapping multi-message token composition.

## Evaluation and deployment

Keep source examples and related variants in one split. Select checkpoints using validation only, then evaluate test data once. Match prompts, decoding limits and trainable final-agent configuration when comparing terminal-only SFT and full training.

At inference, remove surrogate computation, bridge calls, proxy losses and reference policies. Use ordinary generation and text exchange. The included inference test replaces the bridge with an exception-raising function and checks that deployment still runs.

Code-generation benchmarks require an isolated evaluator; this project does not execute arbitrary generated programs. Dataset adapters, benchmark extraction rules and code execution infrastructure are not bundled.
