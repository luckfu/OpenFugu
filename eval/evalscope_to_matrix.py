#!/usr/bin/env python3
"""Convert EvalScope per-sample reviews into OpenFugu router training inputs.

把多个 worker 各自的 EvalScope 输出目录合并为：
1. predictions JSONL —— 与 eval_bfcl.py 输出同构，train_trinity_bfcl.py 可直接 --predictions 消费；
2. worker matrix CSV —— case_id × worker 的 0/1 矩阵，便于人工查看。

关键职责是失败分类：EvalScope 会把 API 超时/限流/欠费等执行失败静默记成 0 分
（BFCL 场景下还会伪装成 ast_decoder:decoder_failed）。这里把执行失败识别出来，
默认整条 case 从矩阵剔除并单独报告，避免把"菜没端上来"当成"厨师不会做"。

用法:
  python eval/evalscope_to_matrix.py \
    --run DeepSeek-V4-Flash=/tmp/evs_flash \
    --run DeepSeek-V4-Pro=/tmp/evs_pro \
    --out-predictions openfugu_bfcl/evalscope_predictions.jsonl \
    --out-matrix openfugu_bfcl/evalscope_worker_matrix.csv

--run 的目录可以是 --work-dir 本身（自动取其中最新时间戳目录），也可以是时间戳目录。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# 执行失败（可重试）的特征：这些不是模型能力问题，不应计 0 分。
TRANSPORT_ERROR_PATTERN = re.compile(
    r"timed? ?out|timeout|rate ?limit|too many requests|connection|"
    r"insufficient_quota|余额不足|无可用资源包|service unavailable|"
    r"internal server error|bad gateway|api.?key|unauthorized|authentication",
    re.IGNORECASE,
)


def find_reviews_dir(run_dir: Path) -> Path:
    """定位 reviews 目录；run_dir 可以是 work-dir 或其中的时间戳目录。"""
    direct = run_dir / "reviews"
    if direct.is_dir():
        return direct
    candidates = sorted(p for p in run_dir.glob("*/reviews") if p.is_dir())
    if not candidates:
        raise SystemExit(f"[evalscope-matrix:error] {run_dir} 下找不到 reviews 目录")
    return candidates[-1]


def classify(score: dict) -> tuple[str, str]:
    """返回 (status, detail)。status: scored / transport_error。"""
    prediction = score.get("prediction")
    if isinstance(prediction, str) and prediction.lstrip().startswith("{"):
        try:
            payload = json.loads(prediction)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and "error" in payload:
            detail = str(payload.get("error"))
            if TRANSPORT_ERROR_PATTERN.search(detail):
                return "transport_error", detail
    meta = score.get("metadata") or {}
    detail = " ".join(str(meta.get(k)) for k in ("error", "error_message"))
    if TRANSPORT_ERROR_PATTERN.search(detail):
        return "transport_error", detail
    return "scored", ""


def main_metric(score: dict, metric: str | None) -> float:
    values = score.get("value") or {}
    if metric:
        if metric not in values:
            raise SystemExit(
                f"[evalscope-matrix:error] 指标 {metric} 不存在，可选: {sorted(values)}"
            )
        return float(values[metric])
    name = score.get("main_score_name")
    if name and name in values:
        return float(values[name])
    if len(values) == 1:
        return float(next(iter(values.values())))
    if "acc" in values:
        return float(values["acc"])
    raise SystemExit(
        f"[evalscope-matrix:error] 无法确定主指标，请用 --metric 指定，可选: {sorted(values)}"
    )


def load_run(worker: str, run_dir: Path, metric: str | None) -> dict[str, dict]:
    """读取一个 worker 的全部 reviews，返回 case_id -> 记录。"""
    rows: dict[str, dict] = {}
    reviews = find_reviews_dir(run_dir)
    for jsonl in sorted(reviews.glob("*/*.jsonl")):
        dataset_tag = jsonl.stem  # 如 bfcl_v4_simple_python / gsm8k_main
        for line in jsonl.open(encoding="utf-8"):
            if not line.strip():
                continue
            entry = json.loads(line)
            ss = entry["sample_score"]
            score = ss["score"]
            meta = ss.get("sample_metadata") or {}
            # BFCL 记录自带官方 id；通用基准回退到 数据集_样本序号。
            case_id = str(meta.get("id") or f"{dataset_tag}_{ss['sample_id']}")
            status, detail = classify(score)
            rows[case_id] = {
                "case_id": case_id,
                "category": str(meta.get("category") or dataset_tag),
                "worker": worker,
                "question": meta.get("question")
                or [[{"role": m.get("role"), "content": m.get("content")}
                     for m in entry.get("messages") or []]],
                "function": meta.get("function") or [],
                "ground_truth": meta.get("ground_truth") or entry.get("target"),
                "prediction": score.get("extracted_prediction") or score.get("prediction"),
                "valid": status == "scored" and main_metric(score, metric) > 0,
                "score": main_metric(score, metric) if status == "scored" else 0.0,
                "status": status,
                "error": detail,
            }
    if not rows:
        raise SystemExit(f"[evalscope-matrix:error] {run_dir} 里没有任何逐样本记录")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="EvalScope 输出 -> OpenFugu 训练矩阵。")
    parser.add_argument(
        "--run", action="append", required=True, metavar="WORKER=DIR",
        help="worker 名称与其 EvalScope 输出目录，可多次指定；顺序即矩阵列顺序",
    )
    parser.add_argument("--metric", help="主指标名（默认取 main_score_name 或 acc）")
    parser.add_argument("--out-predictions", default="openfugu_bfcl/evalscope_predictions.jsonl")
    parser.add_argument("--out-matrix", default="openfugu_bfcl/evalscope_worker_matrix.csv")
    parser.add_argument(
        "--on-failure", choices=["exclude", "zero"], default="exclude",
        help="执行失败的处理：exclude=整条 case 剔除（默认），zero=按 0 分计入",
    )
    args = parser.parse_args()

    runs: list[tuple[str, dict[str, dict]]] = []
    for spec in args.run:
        worker, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"[evalscope-matrix:error] --run 需要 WORKER=DIR 格式: {spec}")
        runs.append((worker, load_run(worker, Path(path).expanduser(), args.metric)))

    workers = [w for w, _ in runs]
    common = set(runs[0][1])
    for _, rows in runs[1:]:
        common &= set(rows)
    union = set().union(*(set(rows) for _, rows in runs))
    if len(common) < len(union):
        print(f"[evalscope-matrix] 警告: {len(union) - len(common)} 条 case 未被所有 worker 覆盖，"
              "已跳过（各 run 的 --datasets/--limit 必须一致）", flush=True)

    kept, dropped = [], defaultdict(list)
    for case_id in sorted(common):
        cells = [rows[case_id] for _, rows in runs]
        failures = [c for c in cells if c["status"] == "transport_error"]
        if failures and args.on_failure == "exclude":
            for cell in failures:
                dropped[case_id].append(f"{cell['worker']}: {cell['error'][:120]}")
            continue
        kept.append((case_id, cells))

    if not kept:
        raise SystemExit("[evalscope-matrix:error] 没有可用 case，全部被执行失败剔除")

    out_pred = Path(args.out_predictions).expanduser()
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    with out_pred.open("w", encoding="utf-8") as f:
        for _, cells in kept:
            for cell in cells:
                record = {k: v for k, v in cell.items() if k not in ("status",)}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_matrix = Path(args.out_matrix).expanduser()
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    with out_matrix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "category", *workers])
        for case_id, cells in kept:
            writer.writerow([case_id, cells[0]["category"], *[f"{c['score']:.6f}" for c in cells]])

    print(f"[evalscope-matrix] worker={workers}", flush=True)
    print(f"[evalscope-matrix] 可用 case={len(kept)} 剔除(执行失败)={len(dropped)}", flush=True)
    for worker, _ in runs:
        scores = [c["score"] for _, cells in kept for c in cells if c["worker"] == worker]
        print(f"[evalscope-matrix] {worker}: {sum(scores):.0f}/{len(scores)}"
              f" = {sum(scores) / len(scores):.2%}", flush=True)
    if dropped:
        print("[evalscope-matrix] 被剔除的 case（对失败 worker 重跑 EvalScope 后重新转换）:",
              file=sys.stderr, flush=True)
        for case_id, reasons in dropped.items():
            for reason in reasons:
                print(f"  {case_id} <- {reason}", file=sys.stderr, flush=True)
    print(f"[evalscope-matrix] predictions -> {out_pred}", flush=True)
    print(f"[evalscope-matrix] matrix      -> {out_matrix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
