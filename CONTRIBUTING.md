# Contributing

Issues and pull requests welcome.

A few ground rules that keep this repo useful:

- **Single-node scope.** This kit targets 1x DGX Spark (GB10). Multi-node/TP=2 proposals
  are out of scope for `main` (DFlash cross-node is unvalidated upstream).
- **Measured claims only.** Performance statements need reproducer + numbers (use the
  `bench/` harness). If you tune a knob, include the profile env and the before/after
  table in the PR.
- **Keep scripts generic.** No hostnames, IPs, usernames, or site-specific paths in
  scripts/docs — everything host-specific is an env var (`SPARK_SSH`, `LAGUNA_HOME`, …).
- **Script changes:** run `bash -n` on every touched `.sh` and
  `python3 -m py_compile bench/ab-bench.py` before submitting. ShellCheck-clean is a plus.
- Don't bump the pins (`VLLM_VERSION`, `FLASHINFER_PIN`) without a tested reason — the
  model card validates vLLM 0.25.1 + FlashInfer 0.6.15.dev20260712 for this model.

Note the repo is MIT but the model is not: `poolside/Laguna-S-2.1-NVFP4` is OpenMDW-1.1 +
the Poolside Acceptable Use Policy.
