#!/usr/bin/env bash
set -euo pipefail

# Colab/GPU entrypoint for training an OpenFugu router from committed
# TRAJECT-Bench evaluation artifacts.
#
# Usage:
#   CONFIG_FILE=configs/trajectbench.example.yaml bash scripts/colab_trajectbench_router.sh

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/trajectbench.example.yaml}"

log() {
  printf '\n\033[1;34m[traject-train]\033[0m %s\n' "$*"
}

cd "$OPENFUGU_DIR"
log "OpenFugu 目录: $OPENFUGU_DIR"
log "配置文件: $CONFIG_FILE"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "未找到 nvidia-smi；没有 GPU 时 Qwen 特征提取会比较慢。"
fi

log "安装训练依赖"
python -m pip install -U pip
python -m pip install -r requirements.txt

if [[ -z "${FUGU_MODEL:-}" ]]; then
  log "解析 Qwen/Qwen3-0.6B 快照路径"
  FUGU_MODEL="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-0.6B"))
PY
)"
  export FUGU_MODEL
fi
log "FUGU_MODEL=$FUGU_MODEL"

log "开始训练 TRAJECT-Bench router"
python train/train_trinity_trajectbench.py \
  --config "$CONFIG_FILE" \
  --router-model "$FUGU_MODEL"

log "完成"
