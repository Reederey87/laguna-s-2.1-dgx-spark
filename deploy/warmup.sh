#!/usr/bin/env bash
# warmup.sh — post-start primer for vllm-laguna.service.
# Runs as ExecStartPost (non-fatal, '-'-prefixed in the unit): waits for /health, then
# primes the paths real traffic hits — plain chat, the poolside_v1 tool parser, and a
# >4K-token prefill — so the first user request doesn't pay JIT/autotune cold costs.
# Safe to run by hand any time:  bash warmup.sh
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-poolside/Laguna-S-2.1-NVFP4}"
HEALTH_WAIT_SECS="${HEALTH_WAIT_SECS:-1500}"

echo "warmup: waiting for $BASE/health (up to ${HEALTH_WAIT_SECS}s)"
deadline=$(( $(date +%s) + HEALTH_WAIT_SECS ))
until curl -sf -m 5 "$BASE/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -lt "$deadline" ] || { echo "warmup: /health never came up — skipping (non-fatal)"; exit 0; }
  sleep 10
done

step() { echo "warmup: $1"; }

step "1/3 plain chat (thinking off)"
curl -fsS -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Say ok.\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"max_tokens\": 8, \"temperature\": 0.7, \"top_p\": 0.95
}" >/dev/null 2>&1 || echo "warmup: chat primer failed (non-fatal)"

step "2/3 tool_choice=auto parser path"
curl -fsS -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Oslo? Use the tool.\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"tools\": [{\"type\": \"function\", \"function\": {
    \"name\": \"get_weather\",
    \"description\": \"Get the current weather for a city\",
    \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}
  }}],
  \"tool_choice\": \"auto\",
  \"max_tokens\": 128, \"temperature\": 0.7, \"top_p\": 0.95
}" >/dev/null 2>&1 || echo "warmup: tool primer failed (non-fatal)"

step "3/3 >4K-token prefill"
prompt=$(python3 -c 'print("Summarize. " + "The quick brown fox jumps over the lazy dog. "*400)')
PROMPT="$prompt" python3 -c 'import json,os; print(json.dumps({
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "chat_template_kwargs": {"enable_thinking": False},
  "max_tokens": 16, "temperature": 0.7, "top_p": 0.95}))' \
  | curl -fsS -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- >/dev/null 2>&1 \
  || echo "warmup: prefill primer failed (non-fatal)"

echo "warmup: done"
