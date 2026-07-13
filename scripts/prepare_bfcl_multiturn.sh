#!/usr/bin/env bash
set -euo pipefail

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/bfcl_multiturn.yaml}"

log() { printf '\n\033[1;34m[bfcl-mt]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[bfcl-mt:error]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$OPENFUGU_DIR"
[[ -f "$CONFIG_FILE" ]] || die "找不到 $CONFIG_FILE；先复制 configs/bfcl_multiturn.example.yaml"

log "安装轻量多轮评测依赖（不安装 torch，不需要 Docker）"
python -m pip install -q litellm pyyaml

IFS=$'\t' read -r CONFIG_BFCL_REPO CONFIG_BFCL_DIR CONFIG_BFCL_REVISION < <(python - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
if p.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}
bfcl = cfg.get("bfcl") or {}
print("\t".join([
    bfcl.get("repo") or "https://github.com/ShishirPatil/gorilla.git",
    bfcl.get("dir") or "gorilla",
    bfcl.get("revision") or "",
]))
PY
)
BFCL_REPO="${BFCL_REPO:-$CONFIG_BFCL_REPO}"
BFCL_DIR="${BFCL_DIR:-$CONFIG_BFCL_DIR}"
BFCL_REVISION="${BFCL_REVISION:-$CONFIG_BFCL_REVISION}"

log "BFCL 目录: $BFCL_DIR"
if [[ ! -d "$BFCL_DIR/.git" ]]; then
  [[ ! -e "$BFCL_DIR" ]] || die "$BFCL_DIR 已存在但不是 git checkout"
  log "浅克隆 Gorilla/BFCL"
  git clone --depth 1 --filter=blob:none --sparse "$BFCL_REPO" "$BFCL_DIR"
  git -C "$BFCL_DIR" sparse-checkout set berkeley-function-call-leaderboard
else
  log "Gorilla/BFCL checkout 已存在"
fi

if [[ -n "$BFCL_REVISION" ]]; then
  if [[ "$(git -C "$BFCL_DIR" rev-parse HEAD)" != "$BFCL_REVISION" ]]; then
    log "切换到固定 BFCL revision: $BFCL_REVISION"
    git -C "$BFCL_DIR" fetch --depth 1 origin "$BFCL_REVISION"
    git -C "$BFCL_DIR" checkout --detach "$BFCL_REVISION"
  else
    log "BFCL revision 已固定: $BFCL_REVISION"
  fi
fi

CMD=(python eval/eval_bfcl_multiturn.py --config "$CONFIG_FILE" --bfcl-dir "$BFCL_DIR")
[[ "${DRY_RUN:-0}" == "1" ]] && CMD+=(--dry-run)
[[ "${NO_RESUME:-0}" == "1" ]] && CMD+=(--no-resume)
[[ "${RETRY_FAILED:-0}" == "1" ]] && CMD+=(--retry-failed)
[[ "${SKIP_PREFLIGHT:-0}" == "1" ]] && CMD+=(--skip-preflight)

log "启动 BFCL Multi-Turn 评测"
"${CMD[@]}"
log "完成"
