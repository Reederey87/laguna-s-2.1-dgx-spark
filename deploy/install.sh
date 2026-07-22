#!/usr/bin/env bash
# install.sh — one-time bare-metal setup for poolside/Laguna-S-2.1-NVFP4 on a single
# NVIDIA DGX Spark (GB10, aarch64). Follows the model card's validated DGX Spark
# recipe: py3.12 + uv venv + vllm==0.25.1 (cu130) + pinned FlashInfer nightly trio
# + ~74 GB of HF weights (model + DFlash draft).
#
# Run ON the Spark as the serving user:  bash install.sh
# Idempotent — safe to re-run; each step skips when already satisfied.
#
# NO sudo required: the card's `apt install python3.12-dev` exists because DGX OS ships no
# Python headers and Triton JIT needs them. We instead use a uv-MANAGED CPython 3.12
# (python-build-standalone), which bundles its own headers — same effect, passwordless.
#
# Why bare-metal, not a container: NGC vllm tags trail upstream (≤0.19 < required
# 0.25.0), and the card's own DGX Spark recipe is the bare-metal uv venv path —
# the only vendor-validated GB10 route for this model.
set -euo pipefail

# --- pins (all from the NVFP4 model card; bump deliberately) -------------------------------
VENV="${VENV:-$HOME/venvs/vllm025}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
FLASHINFER_PIN="${FLASHINFER_PIN:-0.6.15.dev20260712}"
MODEL_ID="${MODEL_ID:-poolside/Laguna-S-2.1-NVFP4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-poolside/Laguna-S-2.1-DFlash-NVFP4}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME

UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || UV="$(command -v uv || true)"

echo "== [1/7] Astral uv"
if [ -z "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="$HOME/.local/bin/uv"
fi
"$UV" --version

echo "== [2/7] uv-managed CPython 3.12 (bundles Python.h — replaces the card's apt python3.12-dev)"
"$UV" python install 3.12

echo "== [3/7] venv at $VENV (managed python 3.12)"
if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv "$VENV" -p 3.12 --managed-python
fi
"$VENV/bin/python" --version
# Headers are what Triton JIT actually needs; verify them in the interpreter's include dir.
"$VENV/bin/python" - <<'PY'
import os, sysconfig, sys
h = os.path.join(sysconfig.get_paths()["include"], "Python.h")
print("Python.h:", h)
sys.exit(0 if os.path.exists(h) else 1)
PY

echo "== [4/7] vllm==$VLLM_VERSION (aarch64 cu130 wheels from PyPI)"
"$UV" pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION" --torch-backend=cu130

echo "== [5/7] FlashInfer nightly trio ==$FLASHINFER_PIN (native NVFP4 path on sm_121;"
echo "    cubin/jit-cache wheels live on the cu130 index; jit-cache pre-seeds kernels so the"
echo "    first serve doesn't JIT the world)"
"$UV" pip install --python "$VENV/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://flashinfer.ai/whl/nightly \
  --extra-index-url https://flashinfer.ai/whl/nightly/cu130 \
  "flashinfer-python==$FLASHINFER_PIN" \
  "flashinfer-cubin==$FLASHINFER_PIN" \
  "flashinfer-jit-cache==$FLASHINFER_PIN"

echo "== [6/7] huggingface_hub + hf_transfer, then weights (~74 GB total)"
"$UV" pip install --python "$VENV/bin/python" "huggingface_hub[hf_transfer]"
# ~74 GB weights + venv; require 120 GB free headroom.
free_kb=$(df --output=avail "$HOME" | tail -n1)
[ "$free_kb" -ge $((120 * 1024 * 1024)) ] || { echo "FAIL: <120 GB free under $HOME" >&2; exit 1; }
# If the repo is gated (OpenMDW click-through), accept the license on HF and export HF_TOKEN.
HF="$VENV/bin/hf"
[ -x "$HF" ] || HF="$VENV/bin/huggingface-cli"
for repo in "$MODEL_ID" "$DFLASH_MODEL_ID"; do
  echo "    downloading $repo ..."
  HF_HUB_ENABLE_HF_TRANSFER=1 "$HF" download "$repo" ${HF_TOKEN:+--token "$HF_TOKEN"}
done

echo "== [7/7] verify imports"
"$VENV/bin/vllm" --version
"$VENV/bin/python" -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo
echo "install done. Next:"
echo "  bash $HOME/laguna-s-2.1/deploy/serve.sh        # foreground first run (~15 min cold start)"
echo "  bash $HOME/laguna-s-2.1/deploy/smoke-test.sh   # from a second shell once /health is up"
