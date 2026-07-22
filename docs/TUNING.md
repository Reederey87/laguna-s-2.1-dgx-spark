# TUNING — every serve parameter, and why

All knobs are environment variables consumed by `deploy/serve.sh` (defaults shown).
Nothing here is speculative: each row is either the model card's explicit instruction or a
measured result from our AB matrix ([PERFORMANCE.md](PERFORMANCE.md)).

## The one flag the card doesn't mention: `MAX_NUM_BATCHED_TOKENS=8192`

**Root cause.** vLLM 0.25.1 chooses default batch sizes in `get_batch_defaults()`
(`vllm/engine/arg_utils.py`): the large-GPU defaults are only picked when
`get_device_total_memory() >= 70 GiB`. On the GB10 the GPU is unified-memory — the device
total reported through that path fails the check, so the engine falls into the *small-GPU*
branch and defaults `max_num_batched_tokens` to **2048**. With DFlash n=15 ×
`--max-num-seqs 32`, the scheduler then reserves speculative headroom and sets
`max_num_scheduled_tokens = 2048 − 448 = 1600` (`vllm/v1/core/sched/scheduler.py` /
`vllm/v1/engine/...` in 0.25.1) — and the engine **warns about exactly this at boot**
("max_num_scheduled_tokens is set to 1600 … Consider increasing max_num_batched_tokens").
Prefill is chunked at 1600 tokens/iteration: a 32K prompt takes ~20 chunks.

Setting `MAX_NUM_BATCHED_TOKENS=8192` ⇒ `max_num_scheduled_tokens=7744` and, measured on
our node (full table in PERFORMANCE.md):

- TTFT p50 **−23% @ 8K prompts** (2,789 → 2,137 ms) and **−13% @ 32K** (11,471 → 9,940 ms)
- decode throughput **unchanged** (c1/c4/c8 within noise)
- cost: KV pool **923,341 → 869,932 tokens (−5.8%)** — still ≈3.3× a full 256K session

This is the default in `serve.sh`. **`MAX_NUM_BATCHED_TOKENS=none` reverts** to passing no
flag (the engine-default 2048 behavior — the a0 baseline).

## Active knobs (serve.sh env vars)

| Env var | Default | Why |
|---|---|---|
| `MODEL_ID` / `DFLASH_MODEL_ID` | `poolside/Laguna-S-2.1-NVFP4` / `-DFlash-NVFP4` | Main weights + DFlash draft, served offline by HF id |
| `NUM_SPEC_TOKENS` | `15` | Card value, AB-verified: n=15 measured **20.9 prose / 43.0 code tok/s** vs n=8's 15.7 / 34.3 at production sampling. n=8 raises per-token acceptance *rate* but lowers absolute accepted length and end-to-end throughput on both content classes |
| `MAX_NUM_SEQS` | `32` | Card: **REQUIRED** — DFlash crashes vLLM at the engine default of 256 (and 256 is exactly what the small-GPU fallback branch picks on GB10). AB'd at 4: rejected (see below) |
| `MAX_MODEL_LEN` | `262144` | 256K as shipped. 1M is native but needs the card's config.json YaRN edit (explicit vendor quality warning; KV pool shrinks proportionally) |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Consensus ceiling on GB10 (vLLM blog, card, community). Don't raise it: upstream vllm#46307 — `profile_run` can exceed the utilization bound on unified memory and wedge the box |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | The fix above. `none` = engine default |
| `LAGUNA_HOST` / `LAGUNA_PORT` | `127.0.0.1` / `8000` | The card uses `0.0.0.0`; loopback unless remote clients need the API. No auth on the endpoint — bind `0.0.0.0` only on a trusted network |
| `GEN_CONFIG_OVERRIDES` | `{"temperature":0.7,"top_p":0.95}` | Card-mandated: clients sending no sampling params degrade on NVFP4 otherwise. **top_k 20 needs no flag** — the checkpoint's own `generation_config.json` already contributes it (verified in boot logs: `{'temperature': 0.7, 'top_k': 20, 'top_p': 0.95, …}`); the override merges |
| `HF_HUB_OFFLINE` | `1` | Weights are pre-pulled by install.sh; serving never touches the network. `0` allows hub lookups |
| `MEM_MARGIN_GIB` | `3` | Headroom for the MemAvailable wait + preflight budget (`util×total + margin`). Guards the request_memory() race that crash-loops GB10 boxes |
| `MEM_WAIT_MAX_ATTEMPTS` / `MEM_WAIT_POLL_SECONDS` | `120` / `5` | Up to 10 min waiting for the margin to clear (e.g. while page cache drains after a previous server stops) |
| `ATTENTION_BACKEND` | unset | Auto already picks `FlashInferBackend` (native decode backend, `arch=sm121`) — set only to force something else |
| `NVCC_THREADS` / `FLASHINFER_NVCC_THREADS` | unset | Optional extra cold-JIT parallelism caps alongside `MAX_JOBS=4`; steady-state irrelevant |

## Environment pins (exported by serve.sh)

- **`CUTE_DSL_ARCH=sm_121a`** — FP4 kernel JIT arch string. Card-required on GB10.
- **`MAX_JOBS=4`** — caps nvcc JIT fan-out. Card warning: uncapped parallel compiles can
  exhaust the 121 GiB unified memory and take the whole box down on a cold cache.
- **`PATH=/usr/local/cuda/bin:$PATH`** — nvcc for JIT.
- **`TRITON_CACHE_DIR` / `FLASHINFER_WORKSPACE_BASE`** under `$LAGUNA_HOME/cache/` —
  **persistent JIT caches.** Without them every restart pays a ~15-minute cold recompile
  (we shipped exactly that bug once: an ephemeral cache location meant `warmup.sh`'s
  priming never survived a container/process recreate). FlashInfer 0.6.15 derives
  `…/.cache/flashinfer/<ver>/<arch>` under `FLASHINFER_WORKSPACE_BASE`
  (`flashinfer/jit/env.py`). Note: a third-party recipe's `FLASHINFER_CACHE_DIR` env var
  **does not exist** in flashinfer 0.6.15 — that recipe works because it mounts the
  default cache path, not because of the variable.
- **`HF_HUB_OFFLINE=1`** (default) — see above.

## Server defaults worth understanding

**Thinking ON by server default** — `--default-chat-template-kwargs '{"enable_thinking":true}'`,
per the base card's agentic recipe. Per-request `chat_template_kwargs
{"enable_thinking": false}` wins. Thinking is emitted in `message.reasoning` by the
`poolside_v1` reasoning parser. **Warning:** thinking burns the output budget — with a tight
`max_tokens` cap the response can truncate to *empty content* (measured: a 512-token
thinking request returned `content=None` at `finish_reason=length`). Give thinking requests
room (the smoke test uses 1024), or pin thinking off per request.

**Tool calls** — `--enable-auto-tool-choice --tool-call-parser poolside_v1
--reasoning-parser poolside_v1`: Poolside's XML protocol. (The `poolside_v1` parser needs
vLLM ≥ the build carrying vllm#47311; fine on 0.25.1. Older builds fall back to `glm47`.)

## Deliberately unset (each a footgun)

- **`--moe-backend` / `--linear-backend`** — auto-selected FlashInferCutlass is correct and
  native on sm_121 with 0.25.1 (confirmed in boot logs). `flashinfer_b12x` is a broken,
  slower opt-in; the generic recipe's `--moe-backend triton` is for B200-class BF16+DFlash,
  not the Spark NVFP4 path. The card's Spark recipe sets neither.
- **`--kv-cache-dtype`** — the checkpoint ships FP8 KV cache (per-tensor minmax),
  auto-detected. No flag needed. (`nvfp4` KV is a community-proven capacity lever on sm_121
  for bigger models if ever needed — not here.)
- **`--max-cudagraph-capture-size`** — 0.25.1's default formula
  (`min(max_num_seqs*(1+spec)*2, 512)` = 512 here) is right. Hardcoding half the engine's
  natural ceiling caused a real ~9–10% concurrency regression in our past; don't pin it.
- **`"method":"dflash"`** in the speculative config — auto-detected; the drafter loads and
  spec metrics flow without it (the card's Spark recipe omits it too).
- **`min_p` / `logit_bias`** — vLLM returns HTTP 400 for these under speculation. Never add
  them to the generation override or client requests.
- **`--dtype bfloat16`** — auto-resolves to bf16 for prefill/decode already.

## Rejected knobs (measured, then dropped)

- **`--max-num-seqs 4`** (a third-party recipe's single-user shape) — **rejected.** c8
  throughput collapses to **37.1 tok/s** with TTFT p50 21.4 s (queue saturation: 4 of 8
  requests wait), and the KV pool actually **shrank to 815,520** — lower than the 8192
  baseline's 869,932, refuting the third-party 926,683-token headline (their measurement
  must come from different memory conditions). Even for a mostly single-user box, seqs=4
  removes all burst headroom (parallel tool calls, subagents).
- **`MAX_NUM_BATCHED_TOKENS=16384`** — **rejected.** Marginal TTFT gain over 8192
  (−0.5% @8K, −4% @32K) for another −9% KV pool (**790,825**, −14% vs engine default) and a
  c4 dip (33.6 vs 38.2 tok/s). A third-party warning that 16384 "can leave no KV blocks" on
  GB10 did not reproduce — it starts fine; the trade is just bad.
- **`NUM_SPEC_TOKENS=8`** — **rejected** (see Active knobs): n=15 wins on both prose and
  code at production sampling.

## Host-level: `vm.min_free_kbytes=2097152`

On GB10 unified memory the kernel low-watermark (~1.25× `min_free_kbytes`) is reserved from
CUDA-visible memory **with no owning PID** — `nvidia-smi` reports it free while vLLM's
`request_memory()` cannot have it. A 4 GiB value on one of our nodes once silently removed
~2.3–2.5 GiB of CUDA-visible memory vs its twin, just enough to fail the 0.85 utilization
budget and crash-loop the server. We run **2097152 (2 GiB)**, runtime and persisted
(`/etc/sysctl.d/90-laguna-oom.conf`); `preflight.sh` warns on runtime-vs-persisted drift so
a reboot can't quietly change memory behavior. Size the reserve deliberately and keep the
two in sync — larger values directly shrink what vLLM can budget.
