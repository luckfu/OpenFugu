#!/usr/bin/env bash
set -euo pipefail

# Download/update TRAJECT-Bench and run OpenFugu's worker evaluation adapter.
#
# Usage:
#   CONFIG_FILE=configs/trajectbench.example.yaml bash scripts/prepare_trajectbench.sh

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/trajectbench.example.yaml}"

log() {
  printf '\n\033[1;34m[trajectbench]\033[0m %s\n' "$*"
}

cd "$OPENFUGU_DIR"
log "OpenFugu 目录: $OPENFUGU_DIR"
log "配置文件: $CONFIG_FILE"

log "安装 TRAJECT-Bench 评测依赖"
python -m pip install -U pip
if [[ "${INSTALL_FULL_REQUIREMENTS:-0}" == "1" || "${INSTALL_FULL_REQUIREMENTS:-false}" == "true" ]]; then
  python -m pip install -r requirements.txt
else
  python -m pip install litellm pyyaml
fi

TRAJECTBENCH_REPO="$(python - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if path.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}
tb = cfg.get("trajectbench") or {}
print(tb.get("repo") or "https://github.com/PengfeiHePower/TRAJECT-Bench.git")
PY
)"

CONFIG_TRAJECTBENCH_DIR="$(python - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if path.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}
tb = cfg.get("trajectbench") or {}
print(tb.get("dir") or "/content/TRAJECT-Bench")
PY
)"
TRAJECTBENCH_DIR="${TRAJECTBENCH_DIR:-$CONFIG_TRAJECTBENCH_DIR}"
if [[ "$TRAJECTBENCH_DIR" == /content/* ]] && [[ ! -w /content ]]; then
  log "配置路径位于 /content，但当前环境不可写；回退到 ./TRAJECT-Bench"
  TRAJECTBENCH_DIR="TRAJECT-Bench"
fi

log "TRAJECT-Bench 目录: $TRAJECTBENCH_DIR"
if [[ ! -d "$TRAJECTBENCH_DIR/.git" ]]; then
  log "克隆 TRAJECT-Bench"
  rm -rf "$TRAJECTBENCH_DIR"
  git clone --depth 1 "$TRAJECTBENCH_REPO" "$TRAJECTBENCH_DIR"
else
  log "TRAJECT-Bench 已存在；尝试拉取最新变更"
  git -C "$TRAJECTBENCH_DIR" pull --ff-only || true
fi

log "开始 TRAJECT-Bench worker 评测"
CMD=(python eval/eval_trajectbench.py --config "$CONFIG_FILE" --trajectbench-dir "$TRAJECTBENCH_DIR")
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN:-false}" == "true" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${NO_RESUME:-0}" == "1" || "${NO_RESUME:-false}" == "true" ]]; then
  CMD+=(--no-resume)
fi
if [[ "${RETRY_FAILED:-0}" == "1" || "${RETRY_FAILED:-false}" == "true" ]]; then
  CMD+=(--retry-failed)
fi
if [[ "${SKIP_PREFLIGHT:-0}" == "1" || "${SKIP_PREFLIGHT:-false}" == "true" ]]; then
  CMD+=(--skip-preflight)
fi

"${CMD[@]}"

log "完成"
