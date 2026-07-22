# DEPLOYMENT — Laguna-S-2.1-NVFP4 on a single DGX Spark

Full walkthrough: prerequisites → install → first serve → smoke gate → systemd service →
day-2 ops. Everything runs **on the Spark** except the rsync/ssh driving, which is done
from a Mac (any SSH client works).

Conventions used throughout:

- The repo lives at **`~/laguna-s-2.1` on the Spark** (the scripts' `LAGUNA_HOME` default).
  Clone it there or rsync it from your Mac.
- `your-spark-host` = your SSH alias/hostname for the Spark. The bench harness reads
  `SPARK_SSH` (default `spark`).
- All commands run as your normal serving user. **No sudo is needed** for the stack itself;
  sudo appears only for optional boot auto-start (`loginctl enable-linger`) and the
  recommended sysctl.

## Prerequisites

| Requirement | Why |
|---|---|
| NVIDIA DGX Spark, GB10 (aarch64), DGX OS / Ubuntu 24.04 | The whole recipe is ARM-native; x86 wheels/images will not work |
| CUDA 13 at `/usr/local/cuda` (nvcc on PATH for JIT) | Triton/FlashInfer compile kernels on first start |
| ~120 GB free disk under `$HOME` | ~74 GB weights + venv + caches |
| Driver ≥ 580.x | 590.48.01 has a unified-memory leak regression; 580.159.03 verified good |
| SSH from your Mac | All ops are driven over SSH |
| Hugging Face access to `poolside/Laguna-S-2.1-NVFP4` + `poolside/Laguna-S-2.1-DFlash-NVFP4` | Possibly gated behind OpenMDW license acceptance — click through on HF and `export HF_TOKEN=...` if the anonymous pull fails |
| Recommended: `vm.min_free_kbytes=2097152` persisted | See below |

**The sysctl.** On this unified-memory platform the kernel low-watermark
(`vm.min_free_kbytes` × ~1.25) is reserved from CUDA-visible memory **with no owning PID** —
`nvidia-smi` shows the memory as free while vLLM cannot have it. We run 2 GiB
(2097152 kB), runtime + persisted:

```bash
echo 'vm.min_free_kbytes=2097152' | sudo tee /etc/sysctl.d/90-laguna-oom.conf
sudo sysctl -w vm.min_free_kbytes=2097152
```

`deploy/preflight.sh` warns if the runtime value drifts from the persisted one (a reboot
would silently change memory behavior otherwise). A *larger* reserve directly shrinks the
memory vLLM can budget at `--gpu-memory-utilization 0.85`.

## 1. Get the repo onto the Spark

```bash
# from the Mac, inside the cloned repo's parent:
rsync -av laguna-s-2.1-dgx-spark/ your-spark-host:~/laguna-s-2.1/
# (or: git clone <this repo> ~/laguna-s-2.1   — on the Spark)
```

## 2. One-time install — `deploy/install.sh`

```bash
ssh -t your-spark-host 'bash ~/laguna-s-2.1/deploy/install.sh'
```

What it does (idempotent; each step skips when already satisfied):

1. Installs [Astral uv](https://astral.sh/uv) to `~/.local/bin` if missing.
2. Installs a **uv-managed CPython 3.12** (python-build-standalone). This bundles `Python.h`,
   which is what Triton JIT actually needs — replacing the model card's
   `sudo apt install python3.12-dev` (DGX OS ships no Python headers). No sudo, passwordless.
3. Creates the venv at `~/venvs/vllm025` on that interpreter; verifies `Python.h` resolves.
4. Installs **`vllm==0.25.1 --torch-backend=cu130`** (aarch64 cu130 wheels are on PyPI).
5. Installs the pinned **FlashInfer nightly trio 0.6.15.dev20260712**
   (`flashinfer-python` / `-cubin` / `-jit-cache`) from `https://flashinfer.ai/whl/nightly{/cu130}/`
   with `--index-strategy unsafe-best-match`. Without `flashinfer-python` the NVFP4 path is
   not native on sm_121; `jit-cache` pre-seeds kernels so the first serve doesn't JIT the world.
6. Pulls the weights (~74 GB total: model + DFlash draft) with `hf_transfer` acceleration,
   after checking 120 GB of free headroom. Honors `HF_TOKEN` if the repos are gated.
7. Verifies `vllm --version` and `import flashinfer`.

Budget ~15 min plus 20–60 min for the weights, depending on bandwidth.

## 3. First serve (foreground) + smoke gate

```bash
# terminal 1 — watch the cold start (~15 min: NVMe weight load + JIT + graph capture)
ssh -t your-spark-host 'bash ~/laguna-s-2.1/deploy/serve.sh'

# terminal 2 — once /health is up (the script waits for it, up to 25 min)
ssh your-spark-host 'bash ~/laguna-s-2.1/deploy/smoke-test.sh'
```

`serve.sh` first runs `preflight.sh` (8 guards: venv present, JIT headers, GPU visible,
weights in the HF cache, memory budget `MemAvailable ≥ util×total + 3 GiB`, sysctl drift,
port free, no co-tenant vLLM/docker server), then waits for the memory margin, then execs
`vllm serve` with the card's flags. Every knob is an env override — see
[TUNING.md](TUNING.md) for the full list and rationale.

`smoke-test.sh` is a 7-check acceptance gate: 1-token canary, short chat, `poolside_v1`
tool-call parser, thinking parser (`enable_thinking` → `message.reasoning`), DFlash
acceptance visible in `/metrics`, concurrency-3, and a ~6K-token prefill probe.
Exit 0 = green.

Two consecutive restarts are worth doing here: the second start must be ≪ 15 min
(~1–2 min), which proves the persistent JIT caches (`TRITON_CACHE_DIR` /
`FLASHINFER_WORKSPACE_BASE` under `~/laguna-s-2.1/cache/`) are working.

## 4. systemd user service + watchdog

```bash
ssh your-spark-host 'mkdir -p ~/.config/systemd/user &&
  cp ~/laguna-s-2.1/deploy/vllm-laguna.service \
     ~/laguna-s-2.1/deploy/vllm-laguna-watchdog.service \
     ~/laguna-s-2.1/deploy/vllm-laguna-watchdog.timer \
     ~/.config/systemd/user/ &&
  systemctl --user daemon-reload &&
  systemctl --user start vllm-laguna.service &&
  systemctl --user start vllm-laguna-watchdog.timer'
```

- **`vllm-laguna.service`** — `ExecStartPre` preflight, `ExecStart` serve.sh,
  `ExecStartPost` warmup (non-fatal), `Restart=on-failure`, `TimeoutStartSec=1800` so
  systemd doesn't kill the 15-min cold start. Binds `127.0.0.1:8000` by default; the unit
  carries a commented `Environment=LAGUNA_HOST=0.0.0.0` line for LAN access (note: the API
  has no auth — only expose it on a trusted network).
- **`vllm-laguna-watchdog.timer`** (5 min) — the "health lies" watchdog: `/health` can
  answer while the engine is wedged, so the canary is a real tagged 1-token completion
  (thinking pinned off). On timeout it triages via `/metrics` — a KV-saturated but
  still-generating engine is *busy, not wedged* and is left alone (with a 3-strike
  livelock backstop); otherwise it restarts `vllm-laguna.service`. It exits silently
  while `/health` is down, so it never fires mid-load or fights a deliberate stop.

**Boot auto-start** (optional, needs sudo once):

```bash
ssh -t your-spark-host 'sudo loginctl enable-linger "$USER" &&
  systemctl --user enable vllm-laguna.service &&
  systemctl --user enable vllm-laguna-watchdog.timer'
```

Linger lets the user manager (and the service) start at boot without an interactive login.
Remember the cold start: the API is up ~15 min after a power-on.

## 5. Day-2 ops

| Task | Command (on the Spark) |
|---|---|
| Logs | `journalctl --user -u vllm-laguna -f` (watchdog: `-u vllm-laguna-watchdog`) |
| Restart | `systemctl --user restart vllm-laguna` (~1–2 min warm) |
| Stop / start | `systemctl --user stop vllm-laguna` / `start` (stop the timer too if you don't want it restarted under you) |
| Status / probe history | `systemctl --user status vllm-laguna`; `cat ~/laguna-s-2.1/deploy/.watchdog-probe.state` |
| Re-run acceptance | `bash ~/laguna-s-2.1/deploy/smoke-test.sh` (read-only, safe any time) |
| Foreground debug run | `systemctl --user stop vllm-laguna vllm-laguna-watchdog.timer && bash ~/laguna-s-2.1/deploy/serve.sh` |

**Update weights / engine:** re-run `install.sh` with overrides —
`VLLM_VERSION=... FLASHINFER_PIN=... bash install.sh` — then restart the service. Weights
are served offline (`HF_HUB_OFFLINE=1` default); a re-pull only happens when you ask for it.

**Roll back a knob:** every serve parameter is an env var in `serve.sh`, so any experiment
reverts by unsetting it (or `MAX_NUM_BATCHED_TOKENS=none` for the engine default) and
restarting. The `bench/profiles/*.env` files document the measured alternatives.

**Update the scripts:** edit locally, rsync the repo over again, `systemctl --user
daemon-reload` if units changed, restart. The bench harness rsyncs `serve.sh` + the bench
client automatically on every run.

**Memory-pressure co-tenants:** unified memory is exclusive — nothing else holding tens of
GiB (another LLM server, a big docker job) can run alongside. Preflight hard-fails on a
second vLLM server and warns on `llama-server`; if vLLM refuses to start with
"free memory less than desired utilization" while `nvidia-smi` looks empty, check the
`vm.min_free_kbytes` reserve first.
