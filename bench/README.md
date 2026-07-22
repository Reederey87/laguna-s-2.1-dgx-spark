# bench/ — AB bench harness (Mac-side, over SSH)

The harness that produced every number in [../docs/PERFORMANCE.md](../docs/PERFORMANCE.md).
A Mac-side orchestrator drives profiles on the Spark over SSH; the bench client itself runs
**on the Spark** so all HTTP is loopback.

## Files

- `ab-run.sh` — Mac-side orchestrator. Per profile: stop `vllm-laguna.service` +
  `vllm-laguna-watchdog.timer` on the Spark → start `../deploy/serve.sh` with the profile
  env (manual, loopback :8000) → wait `/health` → startup facts → warmup → bench →
  teardown. The service + watchdog are restored on exit (EXIT trap), even on
  failure/Ctrl-C.
- `ab-bench.py` — stdlib-only streaming bench client (runs ON the Spark via ssh; JSON to
  stdout, progress to stderr). Sends `chat_template_kwargs {"enable_thinking": false}` so
  thinking can't eat short probes; scrapes `/metrics` around each batch for DFlash
  acceptance deltas and engine-side aggregate tok/s.
- `launch-ab.sh` — on-Spark manual-serve launcher. Exists because
  `ssh host "… nohup … &"` on a compound command keeps ssh's fds open and hangs the
  orchestrator; invoked as a single simple command so ssh returns immediately.
- `profiles/*.env` — one KEY=VAL list per profile; injected via `env` into `serve.sh`
  (every knob is a serve.sh env override; baseline = `MAX_NUM_BATCHED_TOKENS=none`).
- `results/<profile>__<ts>/` — per-run JSON + stderr + startup-facts.txt + warmup.log
  (gitignored).

## Run

```bash
export SPARK_SSH=your-spark-host        # SSH alias/hostname (default: "spark")
bash bench/ab-run.sh a0-baseline        # one profile (~10-15 min)
bash bench/ab-run.sh all                # full matrix (~1.5-2 h)
bash bench/ab-run.sh a1-batched8192 --decode-tokens 1024
```

Prereqs: the deploy tree live on the Spark at `~/laguna-s-2.1` (install.sh done, weights
pulled). The orchestrator rsyncs the latest `serve.sh` / `ab-bench.py` / `launch-ab.sh`
and generates the ~8K/~32K prompt files on every run.

**Production impact:** the Laguna endpoint is down for the duration; the final restore
brings `vllm-laguna.service` + the watchdog timer back.

## Profiles and verdicts

| Profile | Knobs | Verdict |
|---|---|---|
| `a0-baseline` | `MAX_NUM_BATCHED_TOKENS=none` (engine default 2048 on GB10) | Reference. Scheduled tokens stuck at 1600 under DFlash; engine warns |
| `a1-batched8192` | `MAX_NUM_BATCHED_TOKENS=8192` | **ADOPTED** — TTFT −23% @8K / −13% @32K, decode unchanged, KV −5.8% |
| `a3-maxseqs4-batched8192` | `MAX_NUM_SEQS=4` + 8192 | REJECTED — KV pool shrank (815,520; refutes a third-party 926K claim), c8 collapses to 37.1 tok/s |
| `a4-batched16384` | `MAX_NUM_BATCHED_TOKENS=16384` | REJECTED — −14% KV pool, no meaningful TTFT gain over a1 |
| `a5-spec8-batched8192` | `NUM_SPEC_TOKENS=8` + 8192 | REJECTED — prose 15.7 vs 20.9, code 34.3 vs 43.0 tok/s at production sampling; n=15 kept |

## Reading results

Each `decode-c*.json` / `ttft-*.json` carries per-request TTFT/tok/s, batch aggregates
(wall-clock + `/metrics`-derived), and DFlash acceptance deltas
(`_derived_mean_accepted_len`, `_derived_accept_rate`, per-position).
`startup-facts.txt` has the KV pool size, the `max_num_scheduled_tokens` warning
(present = still on the 2048 default), and the CUDA-graph pool.

Compare profiles on: KV pool tokens, TTFT p50 @8K/32K, decode c1/c4/c8 tok/s, mean
accepted length. Mind the caveats in [../docs/PERFORMANCE.md](../docs/PERFORMANCE.md):
KV pool varies ±15% boot-to-boot (compare only within a run), and TTFT p50 blends
prefix-cache-warm reps.

**Sampling matters:** the matrix runs temp 1.0 probes (comparable across engines, but
suppresses DFlash acceptance). For realistic spec-decode numbers use
`ab-bench.py --server-defaults` — no client sampling params, so the server's temp 0.7 /
top_p 0.95 / top_k 20 rule applies (how the a5 probe and the headline 20.9/43.0 tok/s
were measured).
