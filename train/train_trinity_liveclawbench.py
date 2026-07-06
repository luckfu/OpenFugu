#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI.
"""
train_trinity_liveclawbench.py — TRINITY router training on LiveClawBench.

LiveClawBench is not a simple question/answer dataset. It is an agent benchmark
with Docker-backed tasks and an official Harbor verifier. This adapter treats
Harbor's scalar reward.txt as the training reward:

  task     : LiveClawBench task directory + instruction.md
  feature  : real Qwen3-0.6B penultimate hidden state of the task prompt
  workers  : Harbor model ids, e.g. custom/deepseek-chat, openai/gpt-4o
  reward   : Harbor verifier score read from reward.txt, cached per task/worker
  train    : sep-CMA-ES over a bias-free worker-selection head

The expensive part is bounded by task_count x worker_count Harbor runs. CMA
fitness then reuses the cached matrix, so candidate evaluation is cheap.

Example:
  python train/train_trinity_liveclawbench.py \
    --liveclawbench-dir /path/to/LiveClawBench \
    --router-model Qwen/Qwen3-0.6B \
    --slot-models "custom/deepseek-chat,custom/qwen-plus" \
    --ae CUSTOM_BASE_URL="$CUSTOM_BASE_URL" \
    --ae CUSTOM_API_KEY="$CUSTOM_API_KEY" \
    --n-train 8 --iters 12 --precompute-all
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HIDDEN = 1024
HIDDEN_POS = -2


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open("rb") as f:
        return tomllib.load(f)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")[:90] or "run"


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


class Backbone:
    """Real Qwen3-0.6B -> penultimate hidden state of a LiveClawBench task."""

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
        self._cache: dict[str, np.ndarray] = {}

    def feature(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        ids = self.tok(f"user: {text}", return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        with self.torch.no_grad():
            h = self.model.model(**ids).last_hidden_state[0, HIDDEN_POS, :]
        v = h.float().cpu().numpy()
        self._cache[text] = v
        return v


def route(head_vec: np.ndarray, feat: np.ndarray, n_workers: int) -> int:
    W = head_vec.reshape(n_workers, HIDDEN)
    return int(np.argmax(W @ feat))


def load_tasks(root: Path, n_train: int, seed: int, domains: set[str] | None,
               difficulties: set[str] | None, include_regex: str | None) -> list[dict]:
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"LiveClawBench tasks directory not found: {tasks_dir}")

    rows = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        toml_path = task_dir / "task.toml"
        instr_path = task_dir / "instruction.md"
        if not toml_path.exists() or not instr_path.exists():
            continue
        meta = (_load_toml(toml_path).get("metadata") or {})
        domain = str(meta.get("domain") or "")
        difficulty = str(meta.get("difficulty") or "")
        if domains and domain not in domains:
            continue
        if difficulties and difficulty not in difficulties:
            continue
        if include_regex and not re.search(include_regex, task_dir.name):
            continue
        instruction = instr_path.read_text(encoding="utf-8").strip()
        prompt = (
            f"LiveClawBench task: {task_dir.name}\n"
            f"Domain: {domain}\n"
            f"Difficulty: {difficulty}\n"
            f"Instruction:\n{instruction}"
        )
        rows.append({
            "name": task_dir.name,
            "path": str(task_dir.relative_to(root)),
            "domain": domain,
            "difficulty": difficulty,
            "prompt": prompt,
        })

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows[:n_train] if n_train else rows


class HarborScorer:
    def __init__(self, root: Path, workers: list[str], job_root: Path,
                 ae: list[str], ee: list[str], ak: list[str],
                 timeout_multiplier: float, harbor_bin: str, debug: bool):
        self.root = root
        self.workers = workers
        self.job_root = job_root
        self.ae, self.ee, self.ak = ae, ee, ak
        self.timeout_multiplier = timeout_multiplier
        self.harbor_bin = harbor_bin
        self.debug = debug
        self.cache_path = job_root / "score_cache.json"
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.job_root.mkdir(parents=True, exist_ok=True)

    def _save(self):
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.cache_path)

    def score(self, task: dict, worker_id: int) -> float:
        worker = self.workers[worker_id]
        key = f"{task['name']}|{worker}"
        if key in self.cache:
            return float(self.cache[key])

        run_dir = self.job_root / f"{_slug(task['name'])}__w{worker_id}__{_sha(worker)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.harbor_bin, "run",
            "-p", task["path"],
            "-a", "openclaw",
            "-m", worker,
            "-n", "1",
            "-o", str(run_dir),
            "--timeout-multiplier", str(self.timeout_multiplier),
        ]
        for item in self.ae:
            cmd.extend(["--ae", item])
        for item in self.ee:
            cmd.extend(["--ee", item])
        for item in self.ak:
            cmd.extend(["--ak", item])
        if self.debug:
            cmd.append("--debug")

        print(f"[harbor] task={task['name']} worker={worker}", flush=True)
        proc = subprocess.run(cmd, cwd=self.root, text=True, capture_output=True)
        (run_dir / "harbor.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "harbor.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            print(f"   [warn] harbor exited {proc.returncode}; treating score as 0.0", flush=True)
            score = 0.0
        else:
            score = self._read_reward(run_dir)
        self.cache[key] = float(score)
        self._save()
        return float(score)

    @staticmethod
    def _read_reward(run_dir: Path) -> float:
        reward_files = sorted(
            glob.glob(str(run_dir / "**" / "logs" / "verifier" / "reward.txt"), recursive=True),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if not reward_files:
            reward_files = sorted(
                glob.glob(str(run_dir / "**" / "reward.txt"), recursive=True),
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )
        if not reward_files:
            print("   [warn] reward.txt not found; score=0.0", flush=True)
            return 0.0
        try:
            return max(0.0, min(1.0, float(Path(reward_files[0]).read_text().strip())))
        except Exception as e:
            print(f"   [warn] could not parse reward.txt: {e}; score=0.0", flush=True)
            return 0.0


def write_matrix_csv(path: Path, tasks: list[dict], workers: list[str], scores: np.ndarray):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "domain", "difficulty", *workers])
        for i, task in enumerate(tasks):
            w.writerow([task["name"], task["domain"], task["difficulty"], *[f"{x:.4f}" for x in scores[i]]])


def main():
    ap = argparse.ArgumentParser(description="Train a TRINITY router head on LiveClawBench via Harbor rewards.")
    ap.add_argument("--liveclawbench-dir", required=True, help="Path to a checked-out Mosi-AI/LiveClawBench repo")
    ap.add_argument("--router-model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--slot-models", required=True, help="CSV of Harbor model ids, e.g. custom/a,openai/gpt-4o")
    ap.add_argument("--n-train", type=int, default=8, help="Number of LiveClawBench tasks; 0 means all")
    ap.add_argument("--domains", default="", help="Optional pipe-separated domain filter")
    ap.add_argument("--difficulties", default="", help="Optional pipe-separated difficulty filter: easy|medium|hard")
    ap.add_argument("--include-regex", default="", help="Optional regex over task directory names")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--jobs-dir", default="jobs/liveclawbench_router")
    ap.add_argument("--out", default="trinity_liveclawbench.npy")
    ap.add_argument("--matrix-out", default="liveclawbench_scores.csv")
    ap.add_argument("--harbor-bin", default=os.environ.get("HARBOR_BIN", "harbor"))
    ap.add_argument("--timeout-multiplier", type=float, default=1.0)
    ap.add_argument("--ae", action="append", default=[], help="Repeatable Harbor agent env KEY=VALUE")
    ap.add_argument("--ee", action="append", default=[], help="Repeatable Harbor environment env KEY=VALUE")
    ap.add_argument("--ak", action="append", default=[], help="Repeatable Harbor agent kwarg key=value")
    ap.add_argument("--precompute-all", action="store_true", help="Run every task/worker pair before CMA")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    import cma

    root = Path(args.liveclawbench_dir).expanduser().resolve()
    workers = [x.strip() for x in args.slot_models.split(",") if x.strip()]
    if not workers:
        raise ValueError("--slot-models must contain at least one Harbor model id")

    domains = {x.strip() for x in args.domains.split("|") if x.strip()} or None
    difficulties = {x.strip() for x in args.difficulties.split("|") if x.strip()} or None
    tasks = load_tasks(root, args.n_train, args.seed, domains, difficulties, args.include_regex or None)
    if not tasks:
        raise RuntimeError("No LiveClawBench tasks selected")
    print(f"[liveclawbench] selected {len(tasks)} tasks, {len(workers)} workers", flush=True)

    bb = Backbone(args.router_model, device=args.device)
    feats = [bb.feature(t["prompt"]) for t in tasks]
    print(f"[liveclawbench] cached Qwen3-0.6B features dim={feats[0].shape[0]}", flush=True)

    job_root = Path(args.jobs_dir).expanduser().resolve()
    scorer = HarborScorer(root, workers, job_root, args.ae, args.ee, args.ak,
                          args.timeout_multiplier, args.harbor_bin, args.debug)
    scores = np.full((len(tasks), len(workers)), np.nan, dtype=np.float32)

    def get_score(i: int, wid: int) -> float:
        if np.isnan(scores[i, wid]):
            scores[i, wid] = scorer.score(tasks[i], wid)
        return float(scores[i, wid])

    if args.precompute_all:
        for i in range(len(tasks)):
            for wid in range(len(workers)):
                get_score(i, wid)

    def fitness(head_vec: np.ndarray) -> float:
        vals = []
        for i, feat in enumerate(feats):
            wid = route(head_vec, feat, len(workers))
            vals.append(get_score(i, wid))
        return float(np.mean(vals))

    # Optional baselines. With --precompute-all these are exact; otherwise they
    # lazily fill only what training touches.
    dim = len(workers) * HIDDEN
    es = cma.CMAEvolutionStrategy(np.zeros(dim), args.sigma0,
                                  {"seed": args.seed, "verbose": -9, "CMA_diagonal": True})
    best_vec, best_fit = np.zeros(dim), fitness(np.zeros(dim))
    print(f"[baseline] zero-head score={best_fit:.3f}", flush=True)

    for it in range(args.iters):
        cands = es.ask()
        fits = [fitness(c) for c in cands]
        es.tell(cands, [-f for f in fits])
        j = int(np.argmax(fits))
        if fits[j] > best_fit:
            best_fit, best_vec = float(fits[j]), cands[j].copy()
        print(f"[iter {it}] best_score={best_fit:.3f} cache={len(scorer.cache)}", flush=True)

    np.save(args.out, best_vec)
    # Fill the matrix for rows/cols already cached; optionally compute missing
    # entries only when the user requested a complete matrix.
    if args.precompute_all:
        for i in range(len(tasks)):
            for wid in range(len(workers)):
                get_score(i, wid)
    matrix_path = Path(args.matrix_out)
    write_matrix_csv(matrix_path, tasks, workers, np.nan_to_num(scores, nan=-1.0))

    print(f"\n[result] coordinator score={best_fit:.3f}")
    print(f"[result] saved head: {args.out}")
    print(f"[result] saved score matrix: {matrix_path}")
    print("[result] learned routing:")
    for task, feat in zip(tasks, feats):
        wid = route(best_vec, feat, len(workers))
        print(f"  {task['name']:36s} -> {workers[wid]}")


if __name__ == "__main__":
    main()
