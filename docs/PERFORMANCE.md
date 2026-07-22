# PERFORMANCE — measured on 1x DGX Spark (GB10)

All numbers measured on our node: vLLM 0.25.1 (cu130), FlashInfer 0.6.15.dev20260712,
`--gpu-memory-utilization 0.85`, 256K max-model-len, DFlash speculative decoding.
Harness: [`bench/`](../bench/README.md) (Mac-side orchestrator + stdlib-only on-node
streaming bench client). Run date: 2026-07-22.

## Headline numbers (production config = a1 adopted)

| Metric | Value |
|---|---|
| Decode, single stream, production sampling (temp 0.7 / top_p 0.95 / top_k 20, DFlash n=15) | **20.9 tok/s prose · 43.0 tok/s code** |
| Decode aggregate, temp-1.0 probes (512-token decodes) | c1 14.8 · c4 38.2 · c8 56.7 tok/s |
| TTFT p50 | 2,137 ms @ 8K prompt · 9,940 ms @ 32K prompt |
| KV pool | 869,932 tokens (FP8 KV — ≈3.3× a full 256K session) |
| DFlash mean accepted length | 1.36 (prose) · 4.76 (code, 31.7% per-token accept rate) at production sampling |
| DFlash uplift vs no speculation | ≈1.5× prose · ≈3× code (no-spec ceiling ~13–14 tok/s, memory-bandwidth bound) |
| Cold start | ≈15 min first boot · ~1–2 min warm (persistent JIT caches) |

Vendor reference on GB10 (model card): prefill 600–800 tok/s; decode ~15 tok/s prose /
22–24 code with DFlash (2.9–3.1 accepted tokens/step). Our code-class decode (43.0 tok/s)
beats the vendor figure on highly-structured content.

## AB matrix: batch-size knobs (a0 vs a1 vs a3 vs a4)

Decode = aggregate tok/s (512-token probes, thinking off, harness sends temp 1.0);
TTFT = p50 ms over 2 reps of the same prompt; KV pool + scheduled tokens from each boot's
startup log.

| Metric | a0 (engine default 2048) | **a1 (8192) — ADOPTED** | a3 (seqs 4 + 8192) — REJECTED | a4 (16384) — REJECTED |
|---|---|---|---|---|
| decode c1 tok/s | 14.8 | 14.8 | 15.0 | 14.9 |
| decode c4 tok/s | 37.1 | **38.2** | 34.7 | 33.6 |
| decode c8 tok/s | **57.1** | 56.7 | 37.1 ⚠ | 56.1 |
| TTFT p50 @ 8K | 2,789 ms | **2,137 ms (−23%)** | 2,150 ms | 2,127 ms |
| TTFT p50 @ 32K | 11,471 ms | **9,940 ms (−13%)** | 9,570 ms | 9,522 ms |
| KV pool tokens | **923,341** | 869,932 (−5.8%) | 815,520 (−11.7%) | 790,825 (−14.3%) |
| `max_num_scheduled_tokens` | 1,600 (engine warns) | 7,744 | 8,136 | 15,936 |

Verdicts:

- **a1 (`MAX_NUM_BATCHED_TOKENS=8192`) — ADOPT.** Fixes the engine-flagged
  1600-scheduled-tokens defect (root cause: GB10 fails vLLM 0.25.1's ≥70 GiB device-memory
  heuristic → small-GPU 2048 default — see [TUNING.md](TUNING.md)), buys −23%/−13%
  long-prefill TTFT, decode and c8 statistically unchanged, cost −5.8% KV pool.
  Best trade of the matrix.
- **a3 (`--max-num-seqs 4` + 8192 — a third-party recipe's exact shape) — REJECT.**
  c8 collapses to 37.1 tok/s with TTFT p50 21.4 s at c8 (queue saturation: 4 of 8 requests
  wait), and the KV pool *shrank* (815,520 — refutes the third-party 926,683-token @ seqs-4
  headline). The card's "max-num-seqs 32 REQUIRED" stands.
- **a4 (16384) — REJECT.** Marginal TTFT gain over a1 for another −9% KV pool and a c4 dip.
  (A third-party "16384 can leave no KV blocks on GB10" warning did not reproduce —
  it starts fine; the trade is just bad.)
- a2 (seqs-4 alone) — dropped as moot after a3's KV refutation.

## A5: DFlash depth (n=15 vs n=8) at production sampling

Dedicated probe: `ab-bench.py --server-defaults` (no client sampling params → the server's
temp 0.7 / top_p 0.95 / top_k 20 rule), c1, 400-token replies, prose + code prompt classes,
production otherwise on the adopted a1 config.

| | prose tok/s | prose acc. len | code tok/s | code acc. len (rate) |
|---|---|---|---|---|
| **n=15 (card default) — KEPT** | **20.9** | **1.36** | **43.0** | **4.76** (31.7%) |
| n=8 | 15.7 (−25%) | 1.17 | 34.3 (−20%) | 4.15 (51.8%) |

n=8 raises the per-token acceptance *rate* (fewer deep positions wasted: 51.8% vs 31.7% on
code) but lowers absolute accepted length and end-to-end throughput on both content
classes. Poolside's card value is the right one for GB10.

## Measurement method

Per profile, `bench/ab-run.sh` does: stop the systemd service + watchdog timer → start
`deploy/serve.sh` with the profile env (manual, loopback :8000) → wait `/health` → record
startup facts (KV pool size, `max_num_scheduled_tokens`, CUDA-graph pool) → warmup → bench:

- `decode-c1` concurrency 1 × 3 reps, `decode-c4` c4 × 2, `decode-c8` c8 × 2 —
  512-token decodes, thinking pinned off (`enable_thinking: false`), nonce-prefixed prompts.
- `ttft-8k` / `ttft-32k` — c1 × 2 reps of a generated ~8K / ~32K-token prompt,
  `max_tokens 32`. TTFT = time to first *content* token.
- The bench client scrapes `/metrics` before/after each batch and reports DFlash
  acceptance deltas (`vllm:spec_decode_*`, discovered by prefix match) plus
  `/metrics`-derived aggregate tok/s alongside wall-clock figures.

Two sampling regimes, on purpose:

- **AB matrix (temp 1.0, top_p 1.0):** engine-comparable probes. Note this *suppresses*
  DFlash acceptance (mean accepted length 0.75–1.67 in the matrix vs 2.9–3.1 at production
  sampling) — rejection sampling at temp 1.0 wastes drafts. That's why the matrix can't
  judge spec-depth changes and why its absolute decode numbers (14.8 c1) sit below
  production-sampling reality (20.9 prose).
- **Production sampling (`--server-defaults`):** used for the a5 probe and the headline
  prose/code numbers — the server's generation_config rules apply, DFlash acceptance is
  realistic.

## Caveats

- **KV pool varies boot to boot (±15%)** — it depends on memory conditions at engine init
  (page cache, co-tenants, fragmentation). Only compare KV pools within a single AB run,
  never across days.
- **TTFT p50 blends prefix-cache-warm reps** — 2 reps of the same prompt means rep 2 hits
  the prefix cache; relative comparison across profiles is valid, absolute values are not
  "cold TTFT".
- Small samples (2–3 reps per cell) — treat <5% deltas as noise. The a1 TTFT deltas (−23% /
  −13%) and the a3 c8 collapse (57 → 37 tok/s) are far outside noise; the a1 c8 dip
  (57.1 → 56.7) is not.
- Numbers are single-node GB10. They do not transfer to RTX 6000 PRO-class cards (~6× the
  memory bandwidth) — beware third-party "GB10" charts that are 3× above vendor-measured
  GB10 figures.
