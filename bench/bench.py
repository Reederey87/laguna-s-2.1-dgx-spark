#!/usr/bin/env python3
"""Laguna micro-benchmark client. Measures TTFT / tok/s / spec-decode
acceptance against the vLLM API. Run it ON the Spark (copy it there or run
over ssh) so all HTTP calls are loopback and numbers are network-clean.

Stdlib only: urllib, json, threading, time, argparse, uuid, re, statistics.
Emits exactly one JSON object to stdout; all progress/errors go to stderr so
stdout stays parseable by a calling script.

Exit code: 0 unless more than 25% of requests errored.
"""
import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit, urlunsplit

# ---- fixed ~200-word default prompt (deterministic; no external file needed) ----
DEFAULT_PROMPT = (
    "Continue this story for at least 300 words, in a natural narrative style. "
    "Story so far: The data center had gone quiet an hour before the cutover "
    "window opened. Rows of servers hummed at their usual pitch, but the usual "
    "traffic of engineers pacing between racks had thinned to just two people: "
    "Mara, who had written half the failover scripts herself, and Devon, who "
    "was still learning where the emergency stop was for the cooling loop. "
    "They had rehearsed this migration a dozen times in staging, moving "
    "synthetic load between two clusters until the runbook felt like muscle "
    "memory, but production always had a way of surfacing the one edge case "
    "nobody scripted for. Mara watched the dashboard as the first shard began "
    "draining, its connection count ticking down in small, honest increments. "
    "Devon read the checklist aloud, not because Mara needed reminding, but "
    "because saying it out loud made the silence feel less enormous. Outside, "
    "the QSFP cables ran their steady 200 gigabits between racks, carrying "
    "nothing more dramatic than heartbeats and health checks, patient as "
    "the two of them waited for the numbers to line up."
)


def base_root(url: str) -> str:
    """Return scheme://netloc with no path, so /health and /metrics resolve
    regardless of whether --url was given with or without a /v1 suffix."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def completions_url(url: str) -> str:
    return url.rstrip("/") + "/chat/completions"


def metrics_url(url: str) -> str:
    return base_root(url) + "/metrics"


def now() -> float:
    return time.time()


def iso_utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ---- Prometheus text-exposition parsing (minimal hand-rolled parser) ----
_METRIC_LINE = re.compile(r"^(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)\s*$")


def parse_metrics(text: str):
    fam = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            fam.setdefault(name, []).append((labels, float(val)))
        except ValueError:
            continue
    return fam


def fam_total(fam, name):
    return sum(v for _, v in fam.get(name, []))


def scrape(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - metrics scrape is best-effort
        print(f"bench: metrics scrape failed: {exc}", file=sys.stderr)
        return ""


def spec_decode_acceptance(fam_before, fam_after):
    """Discover vllm:spec_decode* families at runtime by prefix match only
    (never hardcode beyond the prefix) and report deltas."""
    names = sorted(
        {n for n in fam_before if n.startswith("vllm:spec_decode")}
        | {n for n in fam_after if n.startswith("vllm:spec_decode")}
    )
    out = {}
    for name in names:
        if "per_pos" in name:
            # Per-position breakdown: label like position="N" or pos="N".
            before_pos, after_pos = {}, {}
            for labels, v in fam_before.get(name, []):
                m = re.search(r'(?:position|pos)="?(\d+)"?', labels)
                if m:
                    before_pos[int(m.group(1))] = before_pos.get(int(m.group(1)), 0.0) + v
            for labels, v in fam_after.get(name, []):
                m = re.search(r'(?:position|pos)="?(\d+)"?', labels)
                if m:
                    after_pos[int(m.group(1))] = after_pos.get(int(m.group(1)), 0.0) + v
            out[name] = {
                str(p): round(after_pos.get(p, 0.0) - before_pos.get(p, 0.0), 3)
                for p in sorted(set(before_pos) | set(after_pos))
            }
        else:
            out[name] = fam_total(fam_after, name) - fam_total(fam_before, name)
    d_acc = out.get("vllm:spec_decode_num_accepted_tokens_total")
    d_draft = out.get("vllm:spec_decode_num_draft_tokens_total")
    d_drafts = out.get("vllm:spec_decode_num_drafts_total")
    out["_derived_accept_rate"] = (d_acc / d_draft) if d_draft else None
    out["_derived_mean_accepted_len"] = (d_acc / d_drafts) if d_drafts else None
    return out


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def run_one_request(url, model, prompt, max_tokens, req_id, results, index,
                    server_defaults=False):
    """Runs in a worker thread; writes its result into results[index] (no
    lock needed — each thread owns a distinct slot)."""
    nonce_prompt = f"[nonce:{uuid.uuid4()}] {prompt}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": nonce_prompt}],
        "max_tokens": max_tokens,
        # Laguna: the thinking kwarg is `enable_thinking` (poolside template).
        # Suppressing it keeps TTFT-to-content valid
        # and stops short probes being eaten by CoT.
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not server_defaults:
        # Default perf-bench behavior: temp 1.0 (deterministic-ish, comparable
        # across engines). --server-defaults instead sends NO sampling params so
        # the server's generation_config rules (temp 0.7/top_p 0.95/top_k 20 for
        # Laguna) — required to measure realistic DFlash acceptance, which
        # collapses under temp 1.0 rejection sampling.
        payload["temperature"] = 1.0
        payload["top_p"] = 1.0
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        completions_url(url), data=body, headers={"Content-Type": "application/json"}
    )
    t0 = now()
    ttft = None
    completion_tokens = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:  # noqa: BLE001 - skip malformed SSE frames
                    continue
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    # TTFT is deliberately the first *content* token only
                    # (not reasoning) — the number a user perceives.
                    if ttft is None and delta.get("content"):
                        ttft = now() - t0
                usage = obj.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens")
    except Exception as exc:  # noqa: BLE001 - network/HTTP errors marked per-request
        err = str(exc)
    wall = now() - t0
    if err is not None:
        results[index] = {"id": req_id, "error": err, "wall_s": round(wall, 4)}
        return
    toks_s = (completion_tokens / wall) if completion_tokens and wall > 0 else None
    results[index] = {
        "id": req_id,
        "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
        "completion_tokens": completion_tokens,
        "wall_s": round(wall, 4),
        "toks_s": round(toks_s, 2) if toks_s is not None else None,
    }


def run_batch(url, model, prompt, max_tokens, concurrency, batch_idx,
              server_defaults=False):
    results = [None] * concurrency
    threads = []
    metrics_before = parse_metrics(scrape(metrics_url(url)))
    batch_t0 = now()
    for i in range(concurrency):
        req_id = f"b{batch_idx}-r{i}"
        t = threading.Thread(
            target=run_one_request,
            args=(url, model, prompt, max_tokens, req_id, results, i,
                  server_defaults),
            daemon=True,
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    batch_wall = now() - batch_t0
    metrics_after = parse_metrics(scrape(metrics_url(url)))

    ok = [r for r in results if r and "error" not in r]
    toks_sum = sum(r["completion_tokens"] or 0 for r in ok)
    agg_toks_wall = (toks_sum / batch_wall) if batch_wall > 0 else 0.0
    d_gen = fam_total(metrics_after, "vllm:generation_tokens_total") - fam_total(
        metrics_before, "vllm:generation_tokens_total"
    )
    agg_toks_metrics = (d_gen / batch_wall) if batch_wall > 0 else 0.0
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]

    batch_summary = {
        "batch": batch_idx,
        "wall_s": round(batch_wall, 4),
        "agg_toks_wall": round(agg_toks_wall, 2),
        "agg_toks_metrics": round(agg_toks_metrics, 2),
        "ttft_p50": round(percentile(ttfts, 0.50), 1) if ttfts else None,
        "ttft_p95": round(percentile(ttfts, 0.95), 1) if ttfts else None,
    }
    return results, batch_summary, metrics_before, metrics_after


def main():
    ap = argparse.ArgumentParser(description="V25-AB micro-benchmark (H-1a)")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="poolside/Laguna-S-2.1-NVFP4")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--reps", type=int, required=True, help="sequential batches")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--server-defaults", action="store_true",
                    help="send no sampling params (server generation_config rules)")
    args = ap.parse_args()

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

    started_at = iso_utc(now())
    per_request = []
    batches = []
    first_metrics_before = None
    last_metrics_after = None

    for b in range(args.reps):
        results, summary, m_before, m_after = run_batch(
            args.url, args.model, prompt, args.max_tokens, args.concurrency, b,
            server_defaults=args.server_defaults,
        )
        if first_metrics_before is None:
            first_metrics_before = m_before
        last_metrics_after = m_after
        per_request.extend(results)
        batches.append(summary)
        print(
            f"bench: batch {b} done — wall={summary['wall_s']}s "
            f"agg_toks_wall={summary['agg_toks_wall']} "
            f"agg_toks_metrics={summary['agg_toks_metrics']}",
            file=sys.stderr,
        )

    ended_at = iso_utc(now())
    acceptance = (
        spec_decode_acceptance(first_metrics_before, last_metrics_after)
        if first_metrics_before is not None
        else {}
    )

    total = len(per_request)
    errored = sum(1 for r in per_request if r and "error" in r)
    error_rate = (errored / total) if total else 0.0

    out = {
        "label": args.label,
        "url": args.url,
        "concurrency": args.concurrency,
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "per_request": per_request,
        "batch": batches,
        "acceptance": acceptance,
        "started_at": started_at,
        "ended_at": ended_at,
        "error_rate": round(error_rate, 4),
    }
    print(json.dumps(out))

    if error_rate > 0.25:
        print(
            f"bench: FAIL — {errored}/{total} requests errored (>25%)",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
