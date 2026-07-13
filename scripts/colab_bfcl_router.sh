#!/usr/bin/env bash
set -euo pipefail

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/bfcl.yaml}"

log() { printf '\n\033[1;34m[bfcl-train]\033[0m %s\n' "$*"; }

cd "$OPENFUGU_DIR"
[[ -f "$CONFIG_FILE" ]] || { echo "找不到 $CONFIG_FILE" >&2; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; fi
log "安装 OpenFugu 训练依赖"
python -m pip install -U pip
python -m pip install -r requirements.txt

if [[ -z "${FUGU_MODEL:-}" ]]; then
  log "下载 Qwen/Qwen3-0.6B"
  FUGU_MODEL="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-0.6B"))
PY
)"
  export FUGU_MODEL
fi

log "训练 BFCL router"
python train/train_trinity_bfcl.py --config "$CONFIG_FILE" --router-model "$FUGU_MODEL"
log "完成"
