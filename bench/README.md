# bench/ — measure your deployment

`bench.py` is the stdlib-only streaming bench client that produced every number in
[../docs/PERFORMANCE.md](../docs/PERFORMANCE.md). Use it to verify your own deployment
reaches the same figures. Run it **on the Spark** so all HTTP is loopback:

```bash
scp bench/bench.py your-spark-host:~/
ssh your-spark-host 'python3 ~/bench.py --help'
```

No dependencies beyond Python 3 (urllib/json/threading). It emits one JSON object to
stdout (per-request TTFT/tok/s, batch aggregates, DFlash acceptance deltas scraped from
`/metrics`); progress and errors go to stderr.

## Typical measurements

```bash
# Single-stream decode, production sampling (the headline 20.9 prose / 43.0 code tok/s):
python3 ~/bench.py --server-defaults --concurrency 1 --reps 3 --max-tokens 400 --label decode-c1

# Batch decode, engine-comparable temp-1.0 probes (the c1/c4/c8 matrix numbers):
python3 ~/bench.py --concurrency 8 --reps 2 --max-tokens 512 --label decode-c8

# TTFT at a long prompt (needs a prompt file; ~8K tokens here):
python3 ~/bench.py --prompt-file prompt-8k.txt --concurrency 1 --reps 2 --max-tokens 32 --label ttft-8k
```

Useful flags: `--url` (default `http://127.0.0.1:8000/v1`), `--model`,
`--max-tokens` (default 2048), `--concurrency`, `--reps`, `--prompt-file`,
`--server-defaults`.

**Sampling matters:** default probes send temp 1.0 (engine-comparable, but suppresses
DFlash acceptance). `--server-defaults` sends no sampling params, so the server's
temp 0.7 / top_p 0.95 / top_k 20 rule applies and spec-decode acceptance is realistic —
that's how the headline numbers were measured. The client pins
`chat_template_kwargs {"enable_thinking": false}` so thinking can't eat short probes.

## Tuning experiments

Every serve parameter is an env override in `../deploy/serve.sh`, so an experiment is:
stop the service → run `VAR=value bash deploy/serve.sh` in the foreground → bench →
restart the service. The measured matrix (what we tried, what won, what was rejected and
why) is documented in [../docs/TUNING.md](../docs/TUNING.md) and
[../docs/PERFORMANCE.md](../docs/PERFORMANCE.md) — read those before re-running rejected
knobs.
