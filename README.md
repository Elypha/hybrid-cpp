# hybrid-cpp

A fork of KoboldCpp focused on fixing pain points specific to hybrid-architecture models (Qwen3.5, Jamba, Falcon-H1, Qwen3-Next, and [others](#supported-architectures)). Upstream KoboldCpp features are preserved unchanged.

### Features

- **[Hybrid partial checkpoints](#hybrid-partial-checkpoints)**: Periodic snapshots of recurrent state during prefill, enabling fast restores on prompt mutations that SmartCache cannot cover. Achieves a 16-40x multi-turn speed-up with < 1% prompt processing (PP) overhead.

## Hybrid Partial Checkpoints

### The Problem

Hybrid/recurrent models carry an SSM (or equivalent) recurrent state in addition to the standard attention KV cache. Existing KV-reuse mechanisms - such as context shifting, fast-forward, and SmartCache - either do not apply to the recurrent half at all, or only snapshot a narrow window near the end of the prompt. Any prompt mutation **deeper than that window** (e.g., SillyTavern World Info inserted at `@depth 4`, chat-history trimming, persona swaps, or lorebook refreshes) forces a full re-prefill from scratch, even when 99% of the token sequence remains identical.

### How It Works

Two CLI flags enable periodic snapshots of the recurrent state during prefill, using llama.cpp's `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` APIs:

```
--hybridcheckpointslots    M   # Keep at most M periodic slots (+1 protective tail slot)
--hybridcheckpointinterval N   # Save every N tokens of prefill (N >= 32)
```
**Note on chunking:** `N` and `--batchsize` must be mutually divisible: if `N < batchsize`, batchsize must be a multiple of `N`; if `N >= batchsize`, `N` must be a multiple of batchsize.

On the next turn, if the new prompt diverges from the previous turn at any depth, the newest slot whose prefix still matches is restored, and only the divergent tail is replayed. No divergence means no save cost wasted - the protective tail slot covers the common "append-only" scenario.

When active, hybrid checkpoints **replace SmartCache** for the hybrid model (SmartCache is auto-disabled). Whilst SmartCache's near-end snapshot only covers the last ~32 tokens, hybrid checkpoints cover the entire context at a configurable granularity, thus SmartCache is redundant.

### Why It Is Effective

- **Constant-size state:** Recurrent state is is cheap - O(1) with respect to token count. The per-slot size depends solely on the model's SSM hidden dimensions and recurrent layer count, not on context length, parameter count, or quantisation. For Qwen3.5-27B: `(n_embd_r=30720 + n_embd_s=786432) × 48 recurrent layers × 4 bytes (F32) = 149.6 MB/slot`.
- **Low save/restore cost:** Saving takes ~2 ms, and restoring takes ~4 ms per slot, negligible compared to prefill times.
- **Deep mutation coverage:** SmartCache only protects the final ~32 tokens; hybrid checkpoints protect the entire context window.
- **Decoupled from batch size:** The decode path plans sub-batches so snapshot cadence is exactly `N` tokens, independent of `--batchsize`.
- **Automatic pool maintenance:** The pool is pruned at each turn boundary to a canonical layout, so restores from arbitrary historical positions never lose more than `interval` tokens.

### Practical defaults

Because per-slot size depends on the model's SSM dimensions rather than parameter count or quantisation, memory profiling is highly predictable. For Qwen3.5-27B, it requires ~150 MB/slot.

Example configurations (Qwen3.5-27B, ~150 MB/slot):

| Context | Flags | RAM (estimated) | Coverage |
|---|---|---|
| 8k | `--hybridcheckpointslots 16 --hybridcheckpointinterval 512` | ~2.3 GB | Full |
| 16k | `--hybridcheckpointslots 32 --hybridcheckpointinterval 256` | ~4.8 GB | bottom 8k |

- See the [benchmark](#benchmark) to understand the trade-offs between interval, slots, and speedup.
- If you are not short of RAM, use `slots * interval >= context length` for full coverage.

### External invalidation

```
POST /api/extra/hybrid_checkpoints/invalidate
```

Clears the hybrid pool. Intended for external tools (e.g. ST plugins) that perform non-monotonic edits such as hide/trim, swipe-delete, branch-graft, and need to force full re-prefill on the next turn.

### Benchmark

All numbers on Qwen3.5-27B Q4_K_M, M4 Max 128 GB, Metal, `I=256 S=64`.

**Multi-turn speedup** (10-turn chat, `@depth 4` insertion each turn, ~15k context):

| | Turn time | Replay tokens |
|---|---|---|
| Vanilla (full re-prefill) | 47--53 s | 8500--8900 |
| Hybrid checkpoint | 1.3--3.1 s | 160--470 |
| **Speedup** | **16--40x** (mean 25x) | - |

**Prompt Processing (PP) throughput overhead** (cold-start, no restore):

| Config | 8k ctx | 16k ctx | 24k ctx | 32k ctx |
|---|---|---|---|---|
| Baseline (hybrid off) | 180.7 t/s | 163.8 t/s | 156.6 t/s | 152.2 t/s |
| I=256, S=64 | 182.9 (+1.2%) | 163.7 (0.0%) | 155.7 (-0.6%) | 150.7 (-1.0%) |
| I=512, S=64 | 184.9 (+2.3%) | 164.9 (+0.7%) | 156.9 (+0.2%) | 152.1 (0.0%) |

Sub-batch splitting overhead is **< 1%** at recommended intervals.

**Restore latency**: save ~2 ms, restore mean **3.6 ms** / max **4.0 ms** (tested across 5 restore points at different context positions).

**Per-slot RAM**: 155-161 MB measured, consistent with the 149.6 MB partial state size (+/- 7%).

### Supported architectures

Only models where `llama_model_is_hybrid` returns true are eligible (defined in [`src/llama-arch.cpp`](src/llama-arch.cpp)):

- Falcon-H1
- Granite-Hybrid
- Jamba
- Kimi-Linear
- LFM2
- LFM2-MoE
- Nemotron-H
- Nemotron-H-MoE
- Plamo2
- Qwen3-Next
- Qwen3.5
- Qwen3.5-MoE

**Pure-recurrent models (Mamba, RWKV) are excluded**: they have no attention KV cache (`mem_attn`) to rebuild prefix state after a partial-only restore, and the `seq_rm` bookkeeping is semantically ill-defined on a position-unaddressable cache. Those models continue to use SmartCache.
