#!/usr/bin/env bash
# ab-run.sh — Mac-side AB orchestrator for Laguna-S-2.1 on a DGX Spark (GB10).
#
#   bash bench/ab-run.sh <profile|all> [--decode-tokens N]
#
# profile = basename of a file in profiles/ without .env (e.g. a0-baseline),
# or `all` = every profiles/*.env in lexical order.
#
# The Spark is reached over SSH; set SPARK_SSH to your SSH alias/hostname
# (default: "spark"). The deploy/ and bench/ trees must live at
# ~/laguna-s-2.1/ on the Spark (install.sh + the systemd unit assume this).
#
# Per profile: stop vllm-laguna.service + watchdog timer on the Spark, start
# deploy/serve.sh with the profile env (manual, loopback :8000), wait /health,
# record KV pool + max_num_scheduled_tokens from the startup log, warmup, then
# bench: decode c1x3 / c4x2 / c8x2 and long-prefill TTFT at ~8K / ~32K tokens.
# Results land in results/<profile>__<timestamp>/{*.json,summary.env}.
#
# On exit (any reason) the previous state is restored: manual serve killed,
# vllm-laguna.service + vllm-laguna-watchdog.timer started again.
#
# Production impact: the Laguna endpoint is down for the whole run except
# during the final service restore — point clients elsewhere during the window.
set -euo pipefail

AB_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(cd "$AB_DIR/../deploy" && pwd)"
RESULTS_DIR="$AB_DIR/results"
mkdir -p "$RESULTS_DIR"

SPARK_SSH="${SPARK_SSH:-spark}"   # SSH alias/hostname of the DGX Spark
NODE_AB="~/laguna-s-2.1/bench"
NODE_DEPLOY="~/laguna-s-2.1/deploy"
HEALTH_URL="http://127.0.0.1:8000/health"
DECODE_TOKENS=512
HEALTH_WAIT_S=1800        # cold start ceiling ~15 min; warm ~2-4 min

PROFILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
    all)
      for f in "$AB_DIR"/profiles/*.env; do PROFILES+=("$(basename "$f" .env)"); done
      shift ;;
    *) PROFILES+=("$1"); shift ;;
  esac
done
[ "${#PROFILES[@]}" -gt 0 ] || { echo "usage: ab-run.sh <profile...|all> [--decode-tokens N]" >&2; exit 1; }
for p in "${PROFILES[@]}"; do
  [ -f "$AB_DIR/profiles/$p.env" ] || { echo "FAIL: profiles/$p.env not found" >&2; exit 1; }
done

log() { echo "[ab-run $(date +%H:%M:%S)] $*"; }

wait_mem_free() {  # preflight needs MemAvailable >= util*total + 3 GiB (~106 GiB)
  local waited=0 avail
  while :; do
    avail="$(ssh "$SPARK_SSH" 'awk "/^MemAvailable:/{print int(\$2/1024/1024)}" /proc/meminfo' | tr -d '\r')"
    [ "${avail:-0}" -ge 106 ] && { log "  MemAvailable ${avail} GiB — clear"; return 0; }
    sleep 5; waited=$((waited+5))
    [ "$waited" -ge 240 ] && { log "WARN: MemAvailable ${avail} GiB still < 106 after 240s"; return 1; }
  done
}

MANUAL_RUNNING=0
kill_manual_serve() {
  # Only pkill when WE started a manual serve — otherwise the pattern would
  # match the production systemd unit's vllm process (2026-07-22 early-abort bug).
  [ "$MANUAL_RUNNING" = "1" ] || return 0
  ssh "$SPARK_SSH" 'pkill -f "vllm serve poolside/Laguna" 2>/dev/null; sleep 5;
        pkill -9 -f "vllm serve poolside/Laguna" 2>/dev/null;
        pkill -f "VLLM::EngineCore" 2>/dev/null; sleep 3;
        pkill -9 -f "VLLM::EngineCore" 2>/dev/null; true' || true
  MANUAL_RUNNING=0
  # wait for the port to actually free so the next profile's preflight passes
  local waited=0
  while ssh "$SPARK_SSH" "curl -fsS -m 3 $HEALTH_URL >/dev/null 2>&1"; do
    sleep 5; waited=$((waited+5))
    [ "$waited" -ge 120 ] && { log "WARN: port 8000 still busy after 120s of teardown"; break; }
  done
  wait_mem_free || true
}

restore_production() {
  log "restoring production: killing manual serve (if any), starting vllm-laguna.service + watchdog timer"
  kill_manual_serve
  # start can race a still-deactivating unit (observed 2026-07-22) — retry until active
  local i
  for i in 1 2 3; do
    ssh "$SPARK_SSH" 'systemctl --user reset-failed vllm-laguna.service 2>/dev/null; systemctl --user start vllm-laguna.service; systemctl --user start vllm-laguna-watchdog.timer' || true
    sleep 5
    [ "$(ssh "$SPARK_SSH" 'systemctl --user is-active vllm-laguna.service' 2>/dev/null | tr -d '\r')" = "active" ] && break
    [ "$(ssh "$SPARK_SSH" 'systemctl --user is-active vllm-laguna.service' 2>/dev/null | tr -d '\r')" = "activating" ] && break
    log "  service not active yet, retry $i/3"
  done
}
trap restore_production EXIT

# ---- one-time node prep: latest scripts + long-prefill prompt files ----------------
log "syncing serve.sh + ab-bench.py + launch-ab.sh to the Spark ($SPARK_SSH)"
ssh "$SPARK_SSH" 'mkdir -p ~/laguna-s-2.1/bench'
rsync -q "$DEPLOY_DIR/serve.sh" "$SPARK_SSH":laguna-s-2.1/deploy/serve.sh
rsync -q "$AB_DIR/ab-bench.py" "$SPARK_SSH":laguna-s-2.1/bench/ab-bench.py
rsync -q "$AB_DIR/launch-ab.sh" "$SPARK_SSH":laguna-s-2.1/bench/launch-ab.sh
ssh "$SPARK_SSH" 'mkdir -p ~/laguna-s-2.1/bench/prompts && python3 - <<"PY"
import os
base = ("The migration runbook described a careful, reversible procedure for moving "
        "stateful services between clusters during a maintenance window, including "
        "health checks, connection draining, and rollback criteria at every step. ")
for name, target_tokens in (("prompt-8k.txt", 8000), ("prompt-32k.txt", 32000)):
    path = os.path.expanduser(f"~/laguna-s-2.1/bench/prompts/{name}")
    words = int(target_tokens / 1.3)
    text = (base * (words // len(base.split()) + 2)).split()[:words]
    with open(path, "w") as f:
        f.write("Summarize the following operational document in detail.\n\n" + " ".join(text))
    print(name, os.path.getsize(path), "bytes")
PY'

run_bench() {  # <profile-outdir> <label> <concurrency> <reps> <max-tokens> [prompt-file]
  local outdir="$1" label="$2" conc="$3" reps="$4" maxtok="$5" pfile="${6:-}"
  local pf_arg=""
  [ -n "$pfile" ] && pf_arg="--prompt-file $NODE_AB/prompts/$pfile"
  log "  bench $label (c$conc x$reps, max_tokens=$maxtok${pfile:+, $pfile})"
  ssh "$SPARK_SSH" "python3 $NODE_AB/ab-bench.py --url http://127.0.0.1:8000/v1 \
    --concurrency $conc --reps $reps --max-tokens $maxtok --label $label $pf_arg" \
    > "$outdir/$label.json" 2> "$outdir/$label.stderr" \
    || log "  WARN: bench $label exited nonzero (see $label.stderr)"
}

wait_health() {
  local waited=0
  while ! ssh "$SPARK_SSH" "curl -fsS -m 5 $HEALTH_URL >/dev/null 2>&1"; do
    sleep 10; waited=$((waited+10))
    if [ "$waited" -ge "$HEALTH_WAIT_S" ]; then
      log "FAIL: /health never came up after ${HEALTH_WAIT_S}s"; return 1
    fi
  done
  log "  /health up after ~${waited}s"
}

for p in "${PROFILES[@]}"; do
  TS="$(date +%Y%m%d-%H%M%S)"
  OUT="$RESULTS_DIR/${p}__${TS}"
  mkdir -p "$OUT"
  cp "$AB_DIR/profiles/$p.env" "$OUT/profile.env"
  log "=== profile $p (results: $OUT)"

  log "  stopping production service + watchdog timer"
  ssh "$SPARK_SSH" 'systemctl --user stop vllm-laguna-watchdog.timer vllm-laguna.service 2>/dev/null; sleep 2; true'
  wait_mem_free || true

  log "  starting manual serve with profiles/$p.env"
  # Build "KEY=VAL KEY=VAL ..." locally from the profile (comments/blanks stripped),
  # pass via env so the remote never needs the profile file itself.
  ENV_VARS="$(grep -vE '^\s*(#|$)' "$AB_DIR/profiles/$p.env" | tr '\n' ' ' || true)"
  log "  env: ${ENV_VARS:-<baseline defaults>}"
  ssh "$SPARK_SSH" "nohup bash $NODE_AB/launch-ab.sh '$p' '$ENV_VARS' > /dev/null 2>&1 < /dev/null &"
  MANUAL_RUNNING=1

  wait_health

  log "  startup facts:"
  ssh "$SPARK_SSH" "grep -E 'GPU KV cache size|Maximum concurrency|max_num_scheduled_tokens|Available KV cache memory|CUDA graph pool memory|init engine' $NODE_AB/serve-$p.log | tail -8" \
    | tee "$OUT/startup-facts.txt"

  log "  warmup"
  ssh "$SPARK_SSH" "bash $NODE_DEPLOY/warmup.sh" > "$OUT/warmup.log" 2>&1 || log "  WARN: warmup nonzero"

  run_bench "$OUT" decode-c1 1 3 "$DECODE_TOKENS"
  run_bench "$OUT" decode-c4 4 2 "$DECODE_TOKENS"
  run_bench "$OUT" decode-c8 8 2 "$DECODE_TOKENS"
  run_bench "$OUT" ttft-8k 1 2 32 prompt-8k.txt
  run_bench "$OUT" ttft-32k 1 2 32 prompt-32k.txt

  log "  stopping manual serve"
  kill_manual_serve
  log "=== profile $p done"
done

log "all profiles done — production restore runs via trap"
