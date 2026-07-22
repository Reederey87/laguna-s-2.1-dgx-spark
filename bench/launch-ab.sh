#!/usr/bin/env bash
# launch-ab.sh — remote (on-Spark) manual-serve launcher for ab-run.sh.
# Exists solely to kill the ssh-hangs-on-backgrounded-compound-command problem:
# `ssh host "cd X && nohup env … serve.sh >log 2>&1 </dev/null &"` backgrounds the
# whole `cd && nohup` list in a subshell that keeps ssh's stdout/stderr open, so
# the local ssh never exits and the orchestrator stalls.
# Invoked as a SINGLE simple command — `nohup bash launch-ab.sh <profile> "<ENV VARS>"
# >/dev/null 2>&1 </dev/null &` — so every fd is redirected and ssh returns at once.
set -eu
PROFILE="${1:?profile name required}"
ENV_VARS="${2:-}"
cd "$HOME/laguna-s-2.1"
LOG="$HOME/laguna-s-2.1/bench/serve-${PROFILE}.log"
# shellcheck disable=SC2086 # intentional word-splitting of "KEY=VAL KEY=VAL"
exec env $ENV_VARS bash "$HOME/laguna-s-2.1/deploy/serve.sh" > "$LOG" 2>&1 < /dev/null
