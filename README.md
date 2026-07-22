# laguna-s-2.1-dgx-spark

Serve [`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
(117.6B MoE / 8.5B active, 256K context, NVFP4, DFlash speculative decoding) with **vLLM on a
single NVIDIA DGX Spark (GB10)** — bare-metal, no sudo, with a systemd user service, an
inference watchdog, and a measured tuning recipe.

**Tested on 1x NVIDIA DGX Spark (GB10); deployed and operated from a Mac over SSH.**
Scope is deliberately single-node: no multi-node/TP=2 content, no multi-model sidecars.

The serving flags follow the model card's own DGX-Spark-validated recipe, plus one measured
fix the card doesn't mention: on GB10, vLLM 0.25.1 picks the *small-GPU* default for
`--max-num-batched-tokens` (2048 → only 1600 scheduled tokens under DFlash) because its
≥70 GiB device-memory heuristic fails on unified memory. Setting it to **8192** buys
**−23% TTFT at 8K / −13% at 32K prompts** for a 5.8% KV-pool cost. Root cause and AB data:
[docs/TUNING.md](docs/TUNING.md), [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Deployment specs

| Component | Value |
|---|---|
| Model | `poolside/Laguna-S-2.1-NVFP4` — 117.6B total / 8.5B active MoE, 256 routed experts top-10, 48 layers (36 sliding-window + 12 global) |
| Weights | ~74 GB total: ~72 GB NVFP4 (routed-expert projections FP4, rest BF16; FP8 KV cache baked into the checkpoint) + ~2.2 GB `poolside/Laguna-S-2.1-DFlash-NVFP4` draft |
| Context | 262,144 (256K) as shipped; 1M native via a config.json edit (vendor quality warning) |
| Engine | vLLM **0.25.1** (`--torch-backend=cu130`, aarch64 PyPI wheels) |
| Kernels | FlashInfer nightly trio **0.6.15.dev20260712** (`flashinfer-python/-cubin/-jit-cache`) |
| Python | uv-managed CPython **3.12** (bundles `Python.h` — no sudo, no `apt install python3.12-dev`) |
| Deployment | Bare-metal uv venv on the Spark; systemd **user** service + 5-min watchdog timer |
| Spec decode | DFlash, 15 speculative tokens (card value — AB-verified against n=8) |
| Hardware | 1x DGX Spark: GB10 Grace Blackwell, aarch64, 121 GiB unified memory, CUDA 13 |

## Measured performance (our node, vLLM 0.25.1 + adopted 8192 profile)

| Metric | Value |
|---|---|
| Decode, single stream (production sampling: temp 0.7 / top_p 0.95 / top_k 20, DFlash n=15) | **20.9 tok/s prose · 43.0 tok/s code** |
| Decode aggregate (512-token probes, temp 1.0) | c1 **14.8** · c4 **38.2** · c8 **56.7** tok/s |
| TTFT p50 | **2,137 ms @ 8K prompt · 9,940 ms @ 32K prompt** |
| KV pool | **~870K tokens** (869,932; FP8 KV, util 0.85 — ≈3.3× a full 256K session) |
| DFlash uplift | ≈**1.5×** prose / ≈**3×** code over the ~13–14 tok/s no-speculation ceiling (memory-bandwidth bound) |
| Cold start | ≈15 min first boot (weights + JIT + graph capture) · ~1–2 min warm (persistent JIT caches) |

Full tables (baseline vs adopted vs rejected profiles, DFlash depth probe, method and caveats):
[docs/PERFORMANCE.md](docs/PERFORMANCE.md). Vendor reference on GB10: prefill 600–800 tok/s,
decode ~15 tok/s prose / 22–24 code with DFlash (2.9–3.1 accepted tokens/step).

## Quickstart

Prerequisites: a DGX Spark on DGX OS / Ubuntu 24.04 with CUDA 13, ~120 GB free disk, SSH
access from your Mac, and HF access to the (possibly gated) model repos.

```bash
# Mac → Spark: put this repo at ~/laguna-s-2.1 on the Spark
rsync -av laguna-s-2.1-dgx-spark/ your-spark-host:~/laguna-s-2.1/

# On the Spark (interactive: apt-free, but the weights pull is ~74 GB)
ssh -t your-spark-host 'bash ~/laguna-s-2.1/deploy/install.sh'     # ~15 min + download
ssh -t your-spark-host 'bash ~/laguna-s-2.1/deploy/serve.sh'       # foreground first run, ~15 min cold start
ssh your-spark-host 'bash ~/laguna-s-2.1/deploy/smoke-test.sh'     # 7-check acceptance gate

# Then run it as a systemd user service (+ watchdog timer)
ssh your-spark-host 'mkdir -p ~/.config/systemd/user &&
  cp ~/laguna-s-2.1/deploy/vllm-laguna.service ~/laguna-s-2.1/deploy/vllm-laguna-watchdog.* ~/.config/systemd/user/ &&
  systemctl --user daemon-reload &&
  systemctl --user start vllm-laguna.service vllm-laguna-watchdog.timer'
```

Full walkthrough incl. boot auto-start (linger), day-2 ops and rollback:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repo map

- `deploy/install.sh` — one-time setup: uv + managed CPython 3.12, vLLM 0.25.1 cu130,
  pinned FlashInfer nightly trio, ~74 GB HF weights pull. Idempotent, no sudo.
- `deploy/preflight.sh` — 8 boot guards (venv, JIT headers, GPU, weights, memory budget,
  sysctl drift, port, co-tenant). `FAIL` refuses to start; `FORCE=1` overrides the two marked guards.
- `deploy/serve.sh` — the hardened serve: card recipe flags + `MAX_NUM_BATCHED_TOKENS=8192`
  default, persistent JIT caches, MemAvailable wait loop, all knobs env-overridable.
- `deploy/smoke-test.sh` — 7-check gate: canary, chat, `poolside_v1` tool-call parser,
  thinking parser, DFlash acceptance, concurrency-3, ~6K-prefill probe.
- `deploy/warmup.sh` — post-start primer (ExecStartPost): chat + tool-call + >4K prefill.
- `deploy/watchdog.sh` + `vllm-laguna-watchdog.{service,timer}` — 5-min inference watchdog:
  tagged 1-token canary, KV-saturation triage (a busy engine is not a wedged engine),
  single-unit restart.
- `deploy/vllm-laguna.service` — systemd user unit (preflight gate, warmup post-start,
  30-min start timeout for the cold boot).
- `bench/` — `bench.py`, the stdlib-only on-node bench client used to produce every
  number in the docs, plus a short guide to measuring your own deployment.
- `docs/` — [DEPLOYMENT](docs/DEPLOYMENT.md) · [TUNING](docs/TUNING.md) ·
  [PERFORMANCE](docs/PERFORMANCE.md).

## License

- **This repo** (scripts + docs): [MIT](LICENSE).
- **The model** is separate: `poolside/Laguna-S-2.1-NVFP4` is **OpenMDW-1.1** (commercial
  use allowed) + the Poolside Acceptable Use Policy — accept it on Hugging Face before pulling.
