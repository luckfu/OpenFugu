#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI.
"""
train_trinity_liveclawbench_offline.py — TRINITY router training from published
LiveClawBench HuggingFace trajectories.

This path does NOT run Harbor or Docker. It reads the official trajectory table:

  Mosi-AI/LiveClawbench-trajectories, split v0.2.1

and builds a cached score matrix:

  task x historical_model -> mean verifier score

Then it trains a bias-free router head over Qwen3-0.6B hidden states to pick the
historical model with the highest expected score.

Use this when Colab cannot run Docker, or when you want a fast offline router
warm start from published leaderboard trajectories. The learned slots correspond
to the historical model names in the dataset, so only use it directly for worker
models that are represented in the trajectory release.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HIDDEN = 1024
HIDDEN_POS = -2


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("YAML config requires PyYAML. Install pyyaml or use a .json config.") from e
    return yaml.safe_load(text) or {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _pipe_list(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return "|".join(str(x) for x in value)


def config_defaults(cfg: dict) -> dict:
    if not cfg:
        return {}
    training = cfg.get("training") or {}
    filters = cfg.get("filters") or {}
    outputs = cfg.get("outputs") or {}
    router = cfg.get("router") or {}
    workers = cfg.get("workers") or []

    models = []
    labels = []
    for w in workers:
        hist = w.get("offline_model_name") or w.get("historical_model_name")
        if hist:
            models.append(str(hist))
            labels.append(str(w.get("name") or hist))

    out = {
        "router_model": router.get("model") or cfg.get("router_model"),
        "device": router.get("device") or training.get("device"),
        "models": ",".join(models) if models else None,
        "slot_labels": ",".join(labels) if labels else None,
        "n_train": training.get("n_train"),
        "iters": training.get("iters"),
        "sigma0": training.get("sigma0"),
        "seed": training.get("seed"),
        "domains": _pipe_list(filters.get("domains")),
        "difficulties": _pipe_list(filters.get("difficulties")),
        "include_regex": filters.get("include_regex"),
        "liveclawbench_dir": cfg.get("liveclawbench_dir"),
        "out": outputs.get("offline_head") or outputs.get("head"),
        "matrix_out": outputs.get("offline_matrix") or outputs.get("matrix"),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [])}


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


def maybe_instruction(root: str | None, case_name: str) -> str | None:
    if not root:
        return None
    p = Path(root).expanduser() / "tasks" / case_name / "instruction.md"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


def task_prompt(row: dict, liveclawbench_dir: str | None) -> str:
    case_name = str(row.get("case_name") or row.get("task_name") or row.get("case_id"))
    instr = maybe_instruction(liveclawbench_dir, case_name)
    parts = [
        f"LiveClawBench task: {case_name}",
        f"Case ID: {row.get('case_id')}",
        f"Domain: {row.get('domain')}",
        f"Difficulty: {row.get('difficulty')}",
        f"Complexity factors: {row.get('complexity_factor')}",
    ]
    if instr:
        parts.append("Instruction:\n" + instr)
    else:
        parts.append("Instruction unavailable; route from task metadata.")
    return "\n".join(parts)


def load_score_matrix(args):
    from datasets import load_dataset

    target_models = [x.strip() for x in args.models.split(",") if x.strip()]
    if not target_models:
        raise SystemExit("--models is required unless config workers include offline_model_name")
    norm_targets = {_norm(m): m for m in target_models}

    domains = {x.strip() for x in args.domains.split("|") if x.strip()} or None
    difficulties = {x.strip() for x in args.difficulties.split("|") if x.strip()} or None

    ds = load_dataset(args.dataset, split=args.split)
    print(f"[offline] loaded {args.dataset} split={args.split} rows={len(ds)}", flush=True)
    print(f"[offline] columns={list(ds.features)}", flush=True)

    # First pass: aggregate score over 3 runs per (case, model).
    by_case: dict[str, dict] = {}
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    available_models = set()

    for row in ds:
        model = str(row.get("model_name") or "")
        available_models.add(model)
        matched = norm_targets.get(_norm(model))
        if not matched:
            continue
        domain = str(row.get("domain") or "")
        difficulty = str(row.get("difficulty") or "")
        case_name = str(row.get("case_name") or "")
        if domains and domain not in domains:
            continue
        if difficulties and difficulty not in difficulties:
            continue
        if args.include_regex and not re.search(args.include_regex, case_name):
            continue
        score = row.get("score")
        if score is None:
            continue
        cid = str(row.get("case_id") or case_name)
        by_case[cid] = dict(row)
        scores[(cid, matched)].append(float(score))

    missing = [m for m in target_models if _norm(m) not in {_norm(x) for x in available_models}]
    if missing:
        print("[offline] warning: requested models not found exactly in dataset:", missing, flush=True)
        print("[offline] available examples:", sorted(available_models)[:40], flush=True)

    tasks = []
    matrix = []
    for cid, row in by_case.items():
        vals = []
        complete = True
        for model in target_models:
            runs = scores.get((cid, model), [])
            if not runs:
                complete = False
                vals.append(np.nan)
            else:
                vals.append(float(np.mean(runs)))
        if args.require_complete and not complete:
            continue
        tasks.append(row)
        matrix.append(vals)

    if args.n_train:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(tasks))[:args.n_train]
        tasks = [tasks[i] for i in order]
        matrix = [matrix[i] for i in order]

    if not tasks:
        raise RuntimeError("No tasks remained after filtering/model matching")

    mat = np.array(matrix, dtype=np.float32)
    if np.isnan(mat).any():
        col_means = np.nanmean(mat, axis=0)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(col_means, inds[1])
    return target_models, tasks, mat


def write_matrix(path: str, tasks: list[dict], models: list[str], mat: np.ndarray):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "case_name", "domain", "difficulty", *models])
        for row, vals in zip(tasks, mat):
            w.writerow([
                row.get("case_id"), row.get("case_name"), row.get("domain"), row.get("difficulty"),
                *[f"{float(x):.4f}" for x in vals],
            ])


def build_parser(defaults: dict | None = None):
    defaults = defaults or {}
    ap = argparse.ArgumentParser(description="Train TRINITY router head from LiveClawBench published trajectories.")
    ap.add_argument("--config", help="YAML/JSON config file; uses workers[*].offline_model_name")
    ap.add_argument("--dataset", default=defaults.get("dataset", "Mosi-AI/LiveClawbench-trajectories"))
    ap.add_argument("--split", default=defaults.get("split", "v0.2.1"))
    ap.add_argument("--models", default=defaults.get("models"),
                    help="CSV of historical model_name values, e.g. GLM-5.2,DeepSeek-V4-Flash")
    ap.add_argument("--slot-labels", default=defaults.get("slot_labels", ""),
                    help="Optional CSV labels for the output slots")
    ap.add_argument("--router-model", default=defaults.get("router_model") or os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--device", default=defaults.get("device"))
    ap.add_argument("--liveclawbench-dir", default=defaults.get("liveclawbench_dir"),
                    help="Optional local LiveClawBench checkout to read task instruction.md")
    ap.add_argument("--n-train", type=int, default=defaults.get("n_train", 0),
                    help="Number of cases to train on; 0 means all complete cases")
    ap.add_argument("--domains", default=defaults.get("domains", ""))
    ap.add_argument("--difficulties", default=defaults.get("difficulties", ""))
    ap.add_argument("--include-regex", default=defaults.get("include_regex", ""))
    ap.add_argument("--iters", type=int, default=defaults.get("iters", 20))
    ap.add_argument("--sigma0", type=float, default=defaults.get("sigma0", 0.3))
    ap.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    ap.add_argument("--out", default=defaults.get("out", "trinity_liveclawbench_offline.npy"))
    ap.add_argument("--matrix-out", default=defaults.get("matrix_out", "liveclawbench_offline_scores.csv"))
    ap.add_argument("--require-complete", action="store_true", default=True,
                    help="Keep only cases with scores for every requested model")
    ap.add_argument("--inspect", action="store_true", help="Print available model names and exit")
    return ap


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)
    args = build_parser(config_defaults(cfg)).parse_args()

    if args.inspect:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split)
        models = sorted(set(str(x) for x in ds["model_name"]))
        print("columns:", list(ds.features))
        print("models:")
        for m in models:
            print(" ", m)
        return 0

    import cma

    models, tasks, mat = load_score_matrix(args)
    labels = [x.strip() for x in args.slot_labels.split(",") if x.strip()] or models
    print(f"[offline] selected cases={len(tasks)} models={models}", flush=True)
    print("[offline] per-model mean: " + ", ".join(
        f"{m}={float(np.mean(mat[:, i])):.3f}" for i, m in enumerate(models)), flush=True)
    write_matrix(args.matrix_out, tasks, labels, mat)
    print(f"[offline] wrote matrix {args.matrix_out}", flush=True)

    bb = Backbone(args.router_model, device=args.device)
    feats = [bb.feature(task_prompt(row, args.liveclawbench_dir)) for row in tasks]
    print(f"[offline] cached Qwen3-0.6B features dim={feats[0].shape[0]}", flush=True)

    n_workers = len(models)
    dim = n_workers * HIDDEN

    def fitness(head_vec: np.ndarray) -> float:
        vals = []
        for feat, row_scores in zip(feats, mat):
            wid = route(head_vec, feat, n_workers)
            vals.append(row_scores[wid])
        return float(np.mean(vals))

    zero = np.zeros(dim)
    base_fit = fitness(zero)
    oracle = float(np.mean(np.max(mat, axis=1)))
    best_single = float(np.max(np.mean(mat, axis=0)))
    print(f"[baseline] zero={base_fit:.3f} best_single={best_single:.3f} oracle={oracle:.3f}", flush=True)

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
        print(f"[iter {it}] best_score={best_fit:.3f} (best_single {best_single:.3f}, oracle {oracle:.3f})", flush=True)

    np.save(args.out, best_vec)
    print(f"\n[result] saved head {args.out}")
    print(f"[result] score={best_fit:.3f} best_single={best_single:.3f} oracle={oracle:.3f}")
    print("[result] learned routing:")
    from collections import Counter
    counts = Counter()
    for i, (row, feat) in enumerate(zip(tasks, feats)):
        wid = route(best_vec, feat, n_workers)
        counts[labels[wid]] += 1
        print(f"  {str(row.get('case_name')):36s} -> {labels[wid]}  score={mat[i, wid]:.3f}")
    print(f"[result] routing distribution: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
