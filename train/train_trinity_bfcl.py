#!/usr/bin/env python3
"""Train a bias-free TRINITY router head from BFCL worker scores."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HIDDEN = 1024
HIDDEN_POS = -2


def load_config(path: str) -> dict:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text) or {}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def router_text(row: dict) -> str:
    messages = []
    for turn in row.get("question") or []:
        if isinstance(turn, list):
            messages.extend(turn)
    functions = []
    for function in row.get("function") or []:
        functions.append({
            "name": function.get("name"),
            "description": function.get("description"),
            "parameters": function.get("parameters"),
        })
    return "\n".join([
        "BFCL function-routing task.",
        f"Category: {row.get('category')}",
        "Conversation:",
        json.dumps(messages, ensure_ascii=False, sort_keys=True),
        "Available functions:",
        json.dumps(functions, ensure_ascii=False, sort_keys=True),
    ])


class Backbone:
    def __init__(self, model_dir: str, device: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.float16 if device and str(device).startswith("cuda") else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.cache: dict[str, np.ndarray] = {}

    def feature(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        tokens = self.tokenizer(
            f"user: {text}", return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        with self.torch.no_grad():
            hidden = self.model.model(**tokens).last_hidden_state[0, HIDDEN_POS, :]
        value = hidden.float().cpu().numpy()
        self.cache[text] = value
        return value


def load_matrix(predictions: Path, workers: list[str]):
    latest = {}
    for row in load_jsonl(predictions):
        latest[(str(row["case_id"]), str(row["worker"]))] = row

    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for (case_id, worker), row in latest.items():
        by_case[case_id][worker] = row

    tasks = []
    matrix = []
    for case_id in sorted(by_case):
        worker_rows = by_case[case_id]
        if any(worker not in worker_rows for worker in workers):
            continue
        representative = worker_rows[workers[0]]
        tasks.append(representative)
        matrix.append([float(worker_rows[worker].get("score") or 0.0) for worker in workers])
    if not tasks:
        raise RuntimeError("没有完整的 BFCL case × worker 评分矩阵")
    return tasks, np.asarray(matrix, dtype=np.float32)


def write_matrix(path: Path, tasks: list[dict], workers: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "category", *workers])
        for task, values in zip(tasks, matrix):
            writer.writerow([task["case_id"], task["category"], *[f"{x:.6f}" for x in values]])


def route(head: np.ndarray, feature: np.ndarray, n_workers: int) -> int:
    return int(np.argmax(head.reshape(n_workers, HIDDEN) @ feature))


def metrics(head, features, matrix, indices, n_workers):
    if not len(indices):
        return float("nan")
    scores = [matrix[i, route(head, features[i], n_workers)] for i in indices]
    return float(np.mean(scores))


def baselines(matrix: np.ndarray, indices: np.ndarray):
    subset = matrix[indices]
    return float(np.max(np.mean(subset, axis=0))), float(np.mean(np.max(subset, axis=1)))


def main() -> int:
    parser = argparse.ArgumentParser(description="从 BFCL 官方评分训练 OpenFugu router head。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--router-model")
    parser.add_argument("--device")
    parser.add_argument("--out")
    parser.add_argument("--matrix-out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outputs = cfg.get("outputs") or {}
    training = cfg.get("training") or {}
    router = cfg.get("router") or {}
    workers = [str(x.get("name") or x["model"]) for x in cfg.get("workers") or []]
    if not workers:
        raise SystemExit("配置中没有 workers")

    predictions = Path(args.predictions or outputs.get("predictions_jsonl") or "openfugu_bfcl/bfcl_predictions.jsonl").expanduser()
    matrix_path = Path(args.matrix_out or outputs.get("matrix_csv") or "openfugu_bfcl/bfcl_worker_matrix.csv").expanduser()
    out_path = Path(args.out or outputs.get("head") or "openfugu_bfcl/trinity_bfcl.npy").expanduser()
    model = args.router_model or router.get("model") or "Qwen/Qwen3-0.6B"
    device = args.device or router.get("device")
    seed = int(training.get("seed") or 42)
    iters = int(training.get("iters") or 20)
    sigma0 = float(training.get("sigma0") or 0.3)

    tasks, matrix = load_matrix(predictions, workers)
    write_matrix(matrix_path, tasks, workers, matrix)
    print(f"[bfcl-train] 完整用例={len(tasks)} worker={workers}", flush=True)
    print("[bfcl-train] 各 worker 平均准确率: " + ", ".join(
        f"{worker}={float(np.mean(matrix[:, i])):.4f}" for i, worker in enumerate(workers)
    ), flush=True)

    backbone = Backbone(model, device)
    features = [backbone.feature(router_text(task)) for task in tasks]
    print(f"[bfcl-train] Qwen 特征已缓存，维度={features[0].shape[0]}", flush=True)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tasks))
    requested = int(training.get("n_train") or 0)
    if requested:
        train_count = min(requested, max(1, len(tasks) - 1))
    else:
        fraction = float(training.get("train_fraction") or 0.8)
        train_count = min(max(1, int(len(tasks) * fraction)), max(1, len(tasks) - 1))
    train_idx = order[:train_count]
    test_idx = order[train_count:]
    if not len(test_idx):
        test_idx = train_idx
    print(f"[bfcl-train] 训练集={len(train_idx)} 验证集={len(test_idx)}", flush=True)

    n_workers = len(workers)
    zero = np.zeros(n_workers * HIDDEN, dtype=np.float64)
    train_single, train_oracle = baselines(matrix, train_idx)
    test_single, test_oracle = baselines(matrix, test_idx)
    print(f"[基线/训练] 最佳单模型={train_single:.4f} 理论上限={train_oracle:.4f}", flush=True)
    print(f"[基线/验证] 最佳单模型={test_single:.4f} 理论上限={test_oracle:.4f}", flush=True)

    import cma
    strategy = cma.CMAEvolutionStrategy(zero, sigma0, {"seed": seed, "verbose": -9, "CMA_diagonal": True})
    best_head = zero.copy()
    best_train = metrics(best_head, features, matrix, train_idx, n_workers)
    for iteration in range(iters):
        candidates = strategy.ask()
        fitness = [metrics(candidate, features, matrix, train_idx, n_workers) for candidate in candidates]
        strategy.tell(candidates, [-x for x in fitness])
        best = int(np.argmax(fitness))
        if fitness[best] > best_train:
            best_train = float(fitness[best])
            best_head = candidates[best].copy()
        heldout = metrics(best_head, features, matrix, test_idx, n_workers)
        print(f"[迭代 {iteration}] 训练={best_train:.4f} 验证={heldout:.4f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, best_head)
    heldout = metrics(best_head, features, matrix, test_idx, n_workers)
    metadata = {
        "workers": workers,
        "hidden_size": HIDDEN,
        "hidden_position": HIDDEN_POS,
        "train_cases": len(train_idx),
        "validation_cases": len(test_idx),
        "train_accuracy": best_train,
        "validation_accuracy": heldout,
        "validation_best_single": test_single,
        "validation_oracle": test_oracle,
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[结果] router head={out_path}", flush=True)
    print(f"[结果] 验证准确率={heldout:.4f} 最佳单模型={test_single:.4f} 理论上限={test_oracle:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
