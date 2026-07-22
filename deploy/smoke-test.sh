#!/usr/bin/env bash
# smoke-test.sh — post-health acceptance gate for the Laguna endpoint.
# Exercises what makes this model's serving stack special: the poolside_v1
# tool/reasoning parsers and DFlash speculative decode. Read-only against the
# server; safe to run any time.
#
#   bash smoke-test.sh                 # waits for /health (default up to 25 min — cold start)
#   BASE=http://127.0.0.1:8000 bash smoke-test.sh
#
# Exit 0 = all hard checks pass. The DFlash-metric check is WARN-only (metric names in
# vLLM 0.25.1 are not yet verified for DFlash).
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-poolside/Laguna-S-2.1-NVFP4}"
HEALTH_WAIT_SECS="${HEALTH_WAIT_SECS:-1500}"

pass=0; failed=0
check() { # check <name> <rc>
  if [ "$2" -eq 0 ]; then echo "PASS: $1"; pass=$((pass+1));
  else echo "FAIL: $1"; failed=$((failed+1)); fi
}
# jqv <python-expr-on-d> — evaluate against JSON on stdin, empty string on any error.
jqv() { python3 -c 'import sys,json
d=json.load(sys.stdin)
try: print('"$1"')
except Exception: pass' 2>/dev/null; }

echo "== waiting for $BASE/health (up to ${HEALTH_WAIT_SECS}s; cold start ≈ 15 min)"
deadline=$(( $(date +%s) + HEALTH_WAIT_SECS ))
until curl -sf -m 5 "$BASE/health" >/dev/null 2>&1; do
  [ "$(date +%s)" -ge "$deadline" ] && { echo "FAIL: /health never came up"; exit 1; }
  sleep 10
done
echo "== /health is up"

# 1. canary ---------------------------------------------------------------------------
resp=$(curl -sS -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"max_tokens\": 1
}" 2>/dev/null)
[ -n "$(echo "$resp" | jqv 'd["choices"][0]["message"]["content"]')" ] || \
  echo "$resp" | python3 -c 'import sys,json; json.load(sys.stdin)["choices"][0]["message"]' >/dev/null 2>&1
check "1-token canary" "$?"

# 2. short chat ------------------------------------------------------------------------
# NOTE: the server defaults enable_thinking=true — every check with a tight
# token budget pins enable_thinking:false so thinking can't eat the budget into empty
# content; check 4 verifies the thinking path itself.
resp=$(curl -sS -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: pong\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"max_tokens\": 16, \"temperature\": 0.7, \"top_p\": 0.95
}" 2>/dev/null)
content=$(echo "$resp" | jqv 'd["choices"][0]["message"].get("content") or ""')
echo "   reply: ${content:0:80}"
echo "$content" | grep -qi pong
check "short chat (expect 'pong')" "$?"

# 3. tool-call parser (poolside_v1) ------------------------------------------------------
resp=$(curl -sS -m 180 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Oslo right now? Use the provided tool.\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"tools\": [{\"type\": \"function\", \"function\": {
    \"name\": \"get_weather\",
    \"description\": \"Get the current weather for a city\",
    \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}
  }}],
  \"tool_choice\": \"auto\",
  \"max_tokens\": 256, \"temperature\": 0.7, \"top_p\": 0.95
}" 2>/dev/null)
ntc=$(echo "$resp" | jqv 'len(d["choices"][0]["message"].get("tool_calls") or [])')
[ -n "$ntc" ] && [ "$ntc" != "0" ]
check "tool_choice=auto emits tool_calls (poolside_v1 parser)" "$?"

# 4. reasoning parser (enable_thinking) --------------------------------------------------
# poolside_v1 emits thinking in message.reasoning (not reasoning_content); accept either.
# 1024 tokens: thinking alone can exceed 512 (truncates at finish=length, empty content).
resp=$(curl -sS -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17*23? Think briefly, then answer.\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": true},
  \"max_tokens\": 1024, \"temperature\": 0.7, \"top_p\": 0.95
}" 2>/dev/null)
rc=$(echo "$resp" | jqv '(d["choices"][0]["message"].get("reasoning_content") or "") + (d["choices"][0]["message"].get("reasoning") or "")')
[ -n "$rc" ]
check "enable_thinking returns reasoning (poolside_v1)" "$?"

# 5. DFlash speculative decode (WARN-only: metric names unverified for DFlash on 0.25.1) --
metrics=$(curl -sS -m 30 "$BASE/metrics" 2>/dev/null)
[ -n "$metrics" ] || { sleep 3; metrics=$(curl -sS -m 30 "$BASE/metrics" 2>/dev/null); }
# herestring, not a pipe: grep -q's early exit would SIGPIPE echo (~111KB > pipe buffer)
# and pipefail would misread that as no-match.
if grep -q '^vllm:spec_decode' <<< "$metrics"; then
  acc=$(echo "$metrics" | awk '/^vllm:spec_decode_num_accepted_tokens_total/{s+=$2} END{printf "%d", s}')
  drafts=$(echo "$metrics" | awk '/^vllm:spec_decode_num_drafts_total/{s+=$2} END{printf "%d", s}')
  echo "   spec_decode: accepted=$acc drafts=$drafts (vendor GB10 reference: 2.9–3.1 accepted tokens/step)"
  [ "${acc:-0}" -gt 0 ]
  check "DFlash acceptance visible in /metrics" "$?"
else
  echo "WARN: no vllm:spec_decode_* series in /metrics — DFlash acceptance unverified (check metric names on 0.25.1)"
fi

# 6. concurrency-3 -----------------------------------------------------------------------
tmp=$(mktemp -d)
for n in 1 2 3; do
  curl -sS -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say the word ok, nothing else. (request $n)\"}],
    \"chat_template_kwargs\": {\"enable_thinking\": false},
    \"max_tokens\": 8, \"temperature\": 0.7, \"top_p\": 0.95
  }" 2>/dev/null >"$tmp/r$n" &
done
wait
cok=1
for n in 1 2 3; do
  [ -n "$(jqv 'd["choices"][0]["message"]' <"$tmp/r$n")" ] || cok=0
done
rm -rf "$tmp"
[ "$cok" = "1" ]
check "concurrency-3 short chats" "$?"

# 7. long-prefill probe (~6K tokens) -------------------------------------------------------
prompt=$(python3 -c 'print("Summarize the following repeated text. " + "The quick brown fox jumps over the lazy dog. "*550)')
t0=$(date +%s)
resp=$(PROMPT="$prompt" python3 -c 'import json,os; print(json.dumps({
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "chat_template_kwargs": {"enable_thinking": False},
  "max_tokens": 32, "temperature": 0.7, "top_p": 0.95}))' \
  | curl -sS -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- 2>/dev/null)
t1=$(date +%s)
[ -n "$(echo "$resp" | jqv 'd["choices"][0]["message"]["content"]')" ]
check "long-prefill ~6K-token request ($((t1-t0))s; vendor prefill ref 600–800 tok/s)" "$?"

echo
echo "smoke: $pass passed, $failed failed"
[ "$failed" -eq 0 ]
