#!/usr/bin/env bash
# serve.sh — hardened single-node serving of poolside/Laguna-S-2.1-NVFP4 on a
# NVIDIA DGX Spark (GB10). Base: the model card's DGX Spark recipe, flag-for-flag.
#
# Hardening on top of the card recipe:
#   * preflight gate (memory budget, sysctl drift, weights, port, co-tenant guard)
#   * MemAvailable wait loop before launch (avoids the request_memory() race — a thin-margin
#     launch is what crash-loops GB10 boxes; also softens upstream vllm#46307 profile_run overrun)
#   * persistent TRITON_CACHE_DIR / FLASHINFER_WORKSPACE_BASE on disk (ephemeral cache =
#     ~15-min cold recompile on every restart)
#   * MAX_JOBS=4 ALWAYS (uncapped nvcc JIT fan-out can exhaust the 121 GiB unified memory
#     and take the whole box down — explicit card warning)
#
# Deliberately NOT set (each a documented footgun — see docs/TUNING.md):
#   * --max-num-seqs stays 32: DFlash crashes vLLM at the default 256 (card: REQUIRED)
#   * no min_p / logit_bias: vLLM 400s them under speculation
#   * no --moe-backend / --linear-backend: auto FlashInferCutlass is correct on sm_121 (0.25.1);
#     flashinfer_b12x is a broken, slower opt-in
#   * no --kv-cache-dtype: the checkpoint ships FP8 KV, auto-detected
#   * no --max-cudagraph-capture-size: 0.25.1's default formula
#     (min(max_num_seqs*(1+spec)*2, 512) = 512 here) is right — hardcoding half the
#     engine's natural ceiling is a measured concurrency regression
#
# --default-chat-template-kwargs enable_thinking:true: server-wide thinking default per
# the base card's agentic recipe. Per-request chat_template_kwargs
# {"enable_thinking": false} still wins. Cost: thinking burns output budget, so tight
# max_tokens caps can truncate to empty content (measured: a 512-token thinking
# request returned content=None at finish=length).
set -euo pipefail

# --- config (env-overridable) --------------------------------------------------------------
LAGUNA_HOME="${LAGUNA_HOME:-$HOME/laguna-s-2.1}"
VENV="${VENV:-$HOME/venvs/vllm025}"
MODEL_ID="${MODEL_ID:-poolside/Laguna-S-2.1-NVFP4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-poolside/Laguna-S-2.1-DFlash-NVFP4}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-15}"          # card value; 2.9–3.1 accepted tokens/step on GB10
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"                # REQUIRED with DFlash (crashes at 256)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"          # 256K as shipped; 1M needs the config.json edit
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LAGUNA_HOST="${LAGUNA_HOST:-127.0.0.1}"           # card uses 0.0.0.0; loopback unless remote clients need it
LAGUNA_PORT="${LAGUNA_PORT:-8000}"
# MAX_NUM_BATCHED_TOKENS: default 8192 — ADOPTED from our AB matrix (see
#   docs/TUNING.md + docs/PERFORMANCE.md): TTFT −23% @8K, −13% @32K, decode
#   unchanged, KV pool −5.8%. Unset/empty/`none` reverts to the vLLM default,
#   which on GB10 falls into the small-GPU heuristic branch
#   (get_device_total_memory < 70 GiB on unified memory) = 2048, and DFlash
#   n=15 × max_num_seqs 32 then leaves only max_num_scheduled_tokens=1600
#   (the engine warns about this). 8192 ⇒ 7744 scheduled. 16384 AB'd worse
#   (−9% more KV, no TTFT gain).
# ATTENTION_BACKEND: auto already picks FlashInferBackend on sm_121 — only
#   set to force something else.
# GEN_CONFIG_OVERRIDES: default mirrors the card; note the checkpoint's own
#   generation_config.json already contributes top_k=20 (verified in logs).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
GEN_CONFIG_OVERRIDES="${GEN_CONFIG_OVERRIDES:-{\"temperature\":0.7,\"top_p\":0.95}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"             # weights pre-pulled by install.sh; 0 = allow hub lookups
MEM_MARGIN_GIB="${MEM_MARGIN_GIB:-3}"
MEM_WAIT_MAX_ATTEMPTS="${MEM_WAIT_MAX_ATTEMPTS:-120}"
MEM_WAIT_POLL_SECONDS="${MEM_WAIT_POLL_SECONDS:-5}"

# --- environment ----------------------------------------------------------------------------
export HF_HOME HF_HUB_OFFLINE
export CUTE_DSL_ARCH=sm_121a                       # FP4 kernel JIT arch string (card-required)
export MAX_JOBS=4                                  # cap nvcc JIT fan-out (card-required)
export PATH="/usr/local/cuda/bin:$PATH"            # nvcc for JIT
export TRITON_CACHE_DIR="$LAGUNA_HOME/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$LAGUNA_HOME/cache/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
# Optional cold-JIT parallelism caps (repo uses NVCC_THREADS=2 + FLASHINFER_NVCC_THREADS=2
# alongside MAX_JOBS=4); only exported when set, so baseline stays unchanged.
[ -n "${NVCC_THREADS:-}" ] && export NVCC_THREADS
[ -n "${FLASHINFER_NVCC_THREADS:-}" ] && export FLASHINFER_NVCC_THREADS

# --- gate + memory wait ----------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  bash "$LAGUNA_HOME/deploy/preflight.sh"
fi

MEMTOTAL_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
NEEDED_KB="$(awk -v t="$MEMTOTAL_KB" -v u="$GPU_MEMORY_UTILIZATION" -v m="$MEM_MARGIN_GIB" \
  'BEGIN{printf "%d", t*u + m*1024*1024}')"
echo "== waiting for MemAvailable >= $((NEEDED_KB/1024/1024)) GiB (util=$GPU_MEMORY_UTILIZATION + ${MEM_MARGIN_GIB}GiB margin)"
for i in $(seq 1 "$MEM_WAIT_MAX_ATTEMPTS"); do
  AVAIL_KB="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
  if [ "$AVAIL_KB" -ge "$NEEDED_KB" ]; then
    echo "== memory margin cleared: $((AVAIL_KB/1024/1024)) GiB available (attempt $i/$MEM_WAIT_MAX_ATTEMPTS)"
    break
  fi
  if [ "$i" -eq "$MEM_WAIT_MAX_ATTEMPTS" ]; then
    echo "== WARNING: margin never cleared after $MEM_WAIT_MAX_ATTEMPTS attempts; proceeding — vLLM's request_memory() will likely fail cleanly"
  fi
  sleep "$MEM_WAIT_POLL_SECONDS"
done

# --- serve (model card's DGX Spark recipe) ---------------------------------------
echo "== starting vllm serve $MODEL_ID on $LAGUNA_HOST:$LAGUNA_PORT"
echo "   first start on a cold cache ≈ 15 min (NVMe load + JIT + graph capture); warm restarts are fast"
EXTRA_ARGS=()
[ -n "$MAX_NUM_BATCHED_TOKENS" ] && [ "$MAX_NUM_BATCHED_TOKENS" != "none" ] \
  && EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[ -n "$ATTENTION_BACKEND" ] && EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
exec "$VENV/bin/vllm" serve "$MODEL_ID" \
  --trust-remote-code \
  --speculative-config "{\"model\":\"$DFLASH_MODEL_ID\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}" \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --override-generation-config "$GEN_CONFIG_OVERRIDES" \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host "$LAGUNA_HOST" --port "$LAGUNA_PORT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
