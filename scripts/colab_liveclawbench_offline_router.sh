#!/usr/bin/env bash
set -euo pipefail

# Colab entrypoint for OFFLINE LiveClawBench router training.
# This does not run Harbor and does not require Docker. It trains from the
# published HuggingFace trajectories:
#   Mosi-AI/LiveClawbench-trajectories
#
# Usage:
#   !git pull
#   !CONFIG_FILE=configs/deepseek_zhipu.example.yaml \
#     bash scripts/colab_liveclawbench_offline_router.sh

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/liveclawbench_colab.example.yaml}"
LIVECLAWBENCH_DIR="${LIVECLAWBENCH_DIR:-}"
LIVECLAWBENCH_REPO="${LIVECLAWBENCH_REPO:-https://github.com/Mosi-AI/LiveClawBench.git}"
CLONE_LIVECLAWBENCH="${CLONE_LIVECLAWBENCH:-1}"

log() {
  printf '\n\033[1;34m[colab-offline]\033[0m %s\n' "$*"
}

cd "$OPENFUGU_DIR"
log "OpenFugu dir: $OPENFUGU_DIR"
log "Config file: $CONFIG_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "nvidia-smi not found; Qwen3-0.6B feature extraction will be slow without GPU."
fi

log "Installing OpenFugu Python dependencies"
python -m pip install -U pip
python -m pip install -r requirements.txt

log "Fetching OpenFugu artifacts"
python scripts/fetch_artifacts.py
export FUGU_VECTOR="$OPENFUGU_DIR/artifacts/model_iter_60.npy"

if [[ -z "${FUGU_MODEL:-}" ]]; then
  log "Resolving Qwen/Qwen3-0.6B snapshot path"
  FUGU_MODEL="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-0.6B"))
PY
)"
  export FUGU_MODEL
fi
log "FUGU_MODEL=$FUGU_MODEL"

if [[ -z "$LIVECLAWBENCH_DIR" ]]; then
  LIVECLAWBENCH_DIR="$(python - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit
text = path.read_text()
if path.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}
print(cfg.get("liveclawbench_dir") or "")
PY
)"
fi

if [[ "$CLONE_LIVECLAWBENCH" == "1" || "$CLONE_LIVECLAWBENCH" == "true" ]]; then
  LIVECLAWBENCH_DIR="${LIVECLAWBENCH_DIR:-/content/LiveClawBench}"
  if [[ ! -d "$LIVECLAWBENCH_DIR/.git" ]]; then
    log "Cloning LiveClawBench for task instruction.md files (no Docker needed)"
    rm -rf "$LIVECLAWBENCH_DIR"
    git clone --depth 1 "$LIVECLAWBENCH_REPO" "$LIVECLAWBENCH_DIR"
  else
    log "LiveClawBench already exists; pulling latest shallow changes if possible"
    git -C "$LIVECLAWBENCH_DIR" pull --ff-only || true
  fi
  LIVECLAW_ARGS=(--liveclawbench-dir "$LIVECLAWBENCH_DIR")
else
  LIVECLAW_ARGS=()
fi

log "Starting offline LiveClawBench router training"
python train/train_trinity_liveclawbench_offline.py \
  --config "$CONFIG_FILE" \
  --router-model "$FUGU_MODEL" \
  "${LIVECLAW_ARGS[@]}"

log "Done"
