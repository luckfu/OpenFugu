#!/usr/bin/env python3
"""Train a TRINITY-style router head from TRAJECT-Bench step samples."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HIDDEN = 1024
HIDDEN_POS = -2


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("YAML config requires PyYAML. Install pyyaml or use a JSON config.") from e
    return yaml.safe_load(text) or {}


def config_defaults(cfg: dict) -> dict:
    if not cfg:
        return {}
    router = cfg.get("router") or {}
    training = cfg.get("training") or {}
    outputs = cfg.get("outputs") or {}
    workers = cfg.get("workers") or []
    return {
        "router_model": router.get("model"),
        "device": router.get("device") or training.get("device"),
        "workers": ",".join(str(w.get("name") or w.get("model")) for w in workers),
        "step_samples": outputs.get("step_samples_jsonl"),
        "out": outputs.get("trajectbench_head") or outputs.get("head"),
        "matrix_out": outputs.get("trajectbench_matrix"),
        "n_train": training.get("n_train"),
        "iters": training.get("iters"),
        "sigma0": training.get("sigma0"),
        "seed": training.get("seed"),
    }


class Backbone:
    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.cache: dict[str, np.ndarray] = {}

    def feature(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        ids = self.tok(f"user: {text}", return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        with self.torch.no_grad():
            h = self.model.model(**ids).last_hidden_state[0, HIDDEN_POS, :]
        v = h.float().cpu().numpy()
        self.cache[text] = v
        return v


def route(head_vec: np.ndarray, feat: np.ndarray, n_workers: int) -> int:
    W = head_vec.reshape(n_workers, HIDDEN)
    return int(np.argmax(W @ feat))


def step_prompt(row: dict) -> str:
    prior = row.get("prior_gold_tools") or []
    parts = [
        "TRAJECT-Bench step-level routing sample.",
        f"Domain: {row.get('domain')}",
        f"Trajectory type: {row.get('trajectory_type')}",
        f"Step index: {row.get('step_index')}",
        "User query:",
        str(row.get("query") or ""),
    ]
    if prior:
        parts.extend([
            "Prior tool calls already planned:",
            json.dumps(prior, ensure_ascii=False, sort_keys=True),
        ])
    else:
        parts.append("Prior tool calls already planned: none")
    return "\n".join(parts)


def load_step_matrix(path: str, worker_names: list[str], require_complete: bool, seed: int, n_train: int):
    rows_by_step: dict[tuple[str, int], dict] = {}
    scores: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)

    with Path(path).expanduser().open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), int(row["step_index"]))
            rows_by_step.setdefault(key, row)
            scores[key][str(row["worker"])] = float(row.get("trajectory_score") or 0.0)

    tasks = []
    matrix = []
    for key, row in rows_by_step.items():
        vals = []
        complete = True
        for worker in worker_names:
            if worker not in scores[key]:
                complete = False
                vals.append(np.nan)
            else:
                vals.append(scores[key][worker])
        if require_complete and not complete:
            continue
        tasks.append(row)
        matrix.append(vals)

    if not tasks:
        raise RuntimeError("No complete step samples found")

    mat = np.array(matrix, dtype=np.float32)
    if np.isnan(mat).any():
        col_means = np.nanmean(mat, axis=0)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(col_means, inds[1])

    if n_train:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(tasks))[:n_train]
        tasks = [tasks[i] for i in order]
        mat = mat[order]
    return tasks, mat


def write_matrix(path: str, tasks: list[dict], workers: list[str], mat: np.ndarray):
    with Path(path).expanduser().open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "step_index", "domain", "trajectory_type", *workers])
        for row, vals in zip(tasks, mat):
            w.writerow([
                row.get("sample_id"), row.get("step_index"), row.get("domain"), row.get("trajectory_type"),
                *[f"{float(x):.6f}" for x in vals],
            ])


def build_parser(defaults: dict):
    ap = argparse.ArgumentParser(description="Train OpenFugu router head from TRAJECT-Bench step samples.")
    ap.add_argument("--config")
    ap.add_argument("--step-samples", default=defaults.get("step_samples") or "openfugu_trajectbench/trajectbench_step_samples.jsonl")
    ap.add_argument("--workers", default=defaults.get("workers"), help="CSV worker names in slot order")
    ap.add_argument("--router-model", default=defaults.get("router_model") or os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--device", default=defaults.get("device"))
    ap.add_argument("--n-train", type=int, default=defaults.get("n_train") or 0)
    ap.add_argument("--iters", type=int, default=defaults.get("iters") or 20)
    ap.add_argument("--sigma0", type=float, default=defaults.get("sigma0") or 0.3)
    ap.add_argument("--seed", type=int, default=defaults.get("seed") or 42)
    ap.add_argument("--out", default=defaults.get("out") or "openfugu_trajectbench/trinity_trajectbench.npy")
    ap.add_argument("--matrix-out", default=defaults.get("matrix_out") or "openfugu_trajectbench/trajectbench_step_matrix.csv")
    ap.add_argument("--allow-incomplete", action="store_true")
    return ap


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)
    args = build_parser(config_defaults(cfg)).parse_args()

    workers = [x.strip() for x in (args.workers or "").split(",") if x.strip()]
    if not workers:
        raise SystemExit("--workers is required, or set workers in config")

    import cma

    tasks, mat = load_step_matrix(args.step_samples, workers, not args.allow_incomplete, args.seed, args.n_train)
    print(f"[traject-train] step contexts={len(tasks)} workers={workers}", flush=True)
    print("[traject-train] per-worker mean: " + ", ".join(
        f"{w}={float(np.mean(mat[:, i])):.4f}" for i, w in enumerate(workers)
    ), flush=True)
    write_matrix(args.matrix_out, tasks, workers, mat)
    print(f"[traject-train] wrote matrix {args.matrix_out}", flush=True)

    bb = Backbone(args.router_model, device=args.device)
    feats = [bb.feature(step_prompt(row)) for row in tasks]
    print(f"[traject-train] cached features dim={feats[0].shape[0]}", flush=True)

    n_workers = len(workers)
    dim = n_workers * HIDDEN

    def fitness(head_vec: np.ndarray) -> float:
        vals = []
        for feat, row_scores in zip(feats, mat):
            wid = route(head_vec, feat, n_workers)
            vals.append(row_scores[wid])
        return float(np.mean(vals))

    zero = np.zeros(dim)
    base_fit = fitness(zero)
    best_single = float(np.max(np.mean(mat, axis=0)))
    oracle = float(np.mean(np.max(mat, axis=1)))
    print(f"[baseline] zero={base_fit:.4f} best_single={best_single:.4f} oracle={oracle:.4f}", flush=True)

    es = cma.CMAEvolutionStrategy(zero, args.sigma0,
                                  {"seed": args.seed, "verbose": -9, "CMA_diagonal": True})
    best_vec, best_fit = zero, base_fit
    for it in range(args.iters):
        cands = es.ask()
        fits = [fitness(c) for c in cands]
        es.tell(cands, [-f for f in fits])
        j = int(np.argmax(fits))
        if fits[j] > best_fit:
            best_fit, best_vec = float(fits[j]), cands[j].copy()
        print(f"[iter {it}] best_score={best_fit:.4f} (best_single {best_single:.4f}, oracle {oracle:.4f})", flush=True)

    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, best_vec)
    print(f"[result] saved head {args.out}")
    print(f"[result] score={best_fit:.4f} best_single={best_single:.4f} oracle={oracle:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
