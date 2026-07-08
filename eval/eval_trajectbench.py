#!/usr/bin/env python3
"""Evaluate worker models on TRAJECT-Bench and export router-training data.

This adapter intentionally uses LiteLLM directly instead of TRAJECT-Bench's
model-provider wrappers, because OpenFugu workers are configured as arbitrary
OpenAI-compatible endpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def load_config(path: str) -> dict:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("YAML config requires PyYAML. Install pyyaml or use JSON.") from e
    return yaml.safe_load(text) or {}


def resolve_secret(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("env:"):
        key = value.split(":", 1)[1]
        val = os.environ.get(key)
        if not val:
            raise RuntimeError(f"environment variable {key} is not set")
        return val
    return value


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "..." + value[-4:]


def norm_tool_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def norm_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return ""
        if re.fullmatch(r"-?\d+", value):
            try:
                return int(value)
            except ValueError:
                pass
        if re.fullmatch(r"-?\d+\.\d+", value):
            try:
                return float(value)
            except ValueError:
                pass
        return re.sub(r"\s+", " ", value)
    if isinstance(value, list):
        return [norm_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): norm_value(v) for k, v in sorted(value.items())}
    return value


def norm_params(tool: dict) -> tuple[tuple[str, Any], ...]:
    params = []
    for key in ("required parameters", "required_parameters", "optional parameters", "optional_parameters"):
        for item in tool.get(key) or []:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            value = norm_value(item.get("value"))
            if value in ("", None):
                continue
            params.append((name, value))
    return tuple(sorted(params, key=lambda x: x[0]))


def clean_tool(tool: dict) -> dict:
    out = {
        "tool name": tool.get("tool name") or tool.get("tool_name") or tool.get("name"),
        "required parameters": tool.get("required parameters") or tool.get("required_parameters") or [],
        "optional parameters": tool.get("optional parameters") or tool.get("optional_parameters") or [],
    }
    if "executed_output" in tool:
        out["executed_output"] = tool["executed_output"]
    return out


def strip_outputs(tool: dict) -> dict:
    out = dict(tool)
    out.pop("executed_output", None)
    out.pop("execution_status", None)
    return out


def extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def parse_prediction(text: str) -> list[dict]:
    obj = extract_json(text)
    tools = obj.get("tool_list") or obj.get("tools") or obj.get("tool list")
    if not isinstance(tools, list):
        raise ValueError("prediction JSON must contain a tool_list array")
    return [clean_tool(x) for x in tools if isinstance(x, dict)]


def classify_error(error: Exception, raw_response: str | None = None) -> str:
    if raw_response == "":
        return "empty_response"
    if isinstance(error, json.JSONDecodeError):
        return "parse_failed"
    text = str(error).lower()
    if "authentication" in text or "身份验证" in text:
        return "auth_failed"
    if "rate limit" in text or "429" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if raw_response is not None:
        return "parse_failed"
    return "call_failed"


def score_prediction(gold: list[dict], pred: list[dict], ordered: bool) -> dict:
    gold_names = [norm_tool_name(x.get("tool name", "")) for x in gold]
    pred_names = [norm_tool_name(x.get("tool name", "")) for x in pred]
    if ordered:
        name_exact = float(gold_names == pred_names)
    else:
        name_exact = float(set(gold_names) == set(pred_names))
    inclusion = len(set(gold_names) & set(pred_names)) / max(1, len(set(gold_names)))

    param_hits = 0
    total = 0
    pred_by_name: dict[str, list[dict]] = {}
    for item in pred:
        pred_by_name.setdefault(norm_tool_name(item.get("tool name", "")), []).append(item)
    for item in gold:
        total += 1
        name = norm_tool_name(item.get("tool name", ""))
        expected = norm_params(item)
        if any(norm_params(candidate) == expected for candidate in pred_by_name.get(name, [])):
            param_hits += 1
    param_accuracy = param_hits / max(1, total)
    score = 0.6 * name_exact + 0.25 * inclusion + 0.15 * param_accuracy
    return {
        "score": round(float(score), 6),
        "name_exact": name_exact,
        "inclusion": round(float(inclusion), 6),
        "param_accuracy": round(float(param_accuracy), 6),
        "gold_tool_count": len(gold),
        "pred_tool_count": len(pred),
    }


def load_tools(base: Path, domain: str) -> Any:
    p = base / "tools" / f"{domain}_tool.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def tool_catalog_for_prompt(tools: Any, max_chars: int = 24000) -> str:
    if tools is None:
        return "No domain tool catalog file was found."
    text = json.dumps(tools, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [tool catalog truncated]"
    return text


def build_prompt(sample: dict, domain: str, trajectory_type: str, tools: Any) -> list[dict]:
    gold_tools = [strip_outputs(clean_tool(x)) for x in sample.get("tool list", [])]
    user = {
        "domain": domain,
        "trajectory_type": trajectory_type,
        "query": sample.get("query"),
        "available_tools": tools,
    }
    system = (
        "You are evaluating tool-use planning. Given a user query and available tools, "
        "predict the complete tool-call trajectory needed to solve the query. "
        "Return only JSON with a top-level key tool_list. Each tool item must use "
        "the keys: tool name, required parameters, optional parameters. Do not include "
        "explanations or markdown."
    )
    # Keep a compact gold-free schema example to stabilize outputs.
    example = {
        "tool_list": [
            {
                "tool name": "Provider: API name",
                "required parameters": [{"name": "parameter_name", "value": "parameter_value"}],
                "optional parameters": [],
            }
        ]
    }
    content = (
        "Task input:\n"
        + json.dumps(user, ensure_ascii=False, indent=2)
        + "\n\nOutput JSON schema example:\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
        + "\n\nGold tool count hint: "
        + str(len(gold_tools))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


@contextmanager
def worker_env(worker: dict):
    updates = {}
    for k, v in (worker.get("env") or {}).items():
        updates[str(k)] = str(resolve_secret(v))
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def call_worker(worker: dict, messages: list[dict], temperature: float, max_tokens: int, timeout: float | None = None) -> str:
    import litellm

    kwargs = {
        "model": worker["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if timeout:
        kwargs["timeout"] = timeout
    api_base = worker.get("api_base") or worker.get("base_url")
    api_key = worker.get("api_key")
    if api_base:
        kwargs["api_base"] = resolve_secret(api_base)
    if api_key:
        kwargs["api_key"] = resolve_secret(api_key)
    with worker_env(worker):
        res = litellm.completion(**kwargs)
    return res.choices[0].message.content or ""


def preflight_workers(workers: list[dict], strict: bool) -> None:
    print("[trajectbench] preflight worker credentials", flush=True)
    for worker in workers:
        name = str(worker.get("name") or worker.get("model"))
        api_key_ref = worker.get("api_key")
        try:
            api_key = resolve_secret(api_key_ref) if api_key_ref else ""
            shown = mask_secret(str(api_key)) if api_key else "<none>"
            print(f"[trajectbench]   {name}: api_key={shown}", flush=True)
        except Exception as e:
            msg = f"[trajectbench:preflight] {name}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            print(msg, flush=True)
            continue

        if not strict:
            continue

        messages = [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": 'Return {"ok": true}.'},
        ]
        try:
            text = call_worker(worker, messages, temperature=0, max_tokens=32, timeout=30)
            print(f"[trajectbench]   {name}: live check ok ({text[:60]!r})", flush=True)
        except Exception as e:
            raise RuntimeError(f"[trajectbench:preflight] {name}: live check failed: {e}") from e


def progress_line(done: int, total: int, ok: int, failed: int, skipped: int = 0) -> str:
    pct = 100.0 * done / max(1, total)
    remaining = max(0, total - done)
    return (
        f"[trajectbench] progress {done}/{total} ({pct:5.1f}%) "
        f"ok={ok} fail={failed} skipped={skipped} remaining={remaining}"
    )


def evaluate_one(
    item: dict,
    worker: dict,
    messages: list[dict],
    gold: list[dict],
    ordered: bool,
    temperature: float,
    max_tokens: int,
    request_timeout: float | None,
) -> dict:
    name = str(worker.get("name") or worker.get("model"))
    row = {
        "sample_id": item["sample_id"],
        "worker": name,
        "model": worker.get("model"),
        "domain": item["domain"],
        "trajectory_type": item["trajectory_type"],
        "trajectory_file": item["trajectory_file"],
        "index": item["index"],
        "query": item["sample"].get("query"),
        "gold_tool_list": gold,
    }
    started = time.time()
    try:
        text = call_worker(worker, messages, temperature, max_tokens, timeout=request_timeout)
        row["raw_response"] = text
        pred = parse_prediction(text)
        metrics = score_prediction(gold, pred, ordered=ordered)
        row.update(metrics)
        row["pred_tool_list"] = pred
    except Exception as e:
        row.update({
            "score": 0.0,
            "name_exact": 0.0,
            "inclusion": 0.0,
            "param_accuracy": 0.0,
            "gold_tool_count": len(gold),
            "pred_tool_count": 0,
            "pred_tool_list": [],
            "error_type": classify_error(e, row.get("raw_response")),
            "error": str(e),
        })
    row["latency_sec"] = round(time.time() - started, 3)
    return row


def iter_samples(base: Path, cfg: dict):
    ev = cfg.get("evaluation") or {}
    domains = ev.get("domains") or []
    traj_types = ev.get("trajectory_types") or ["parallel"]
    traj_files = ev.get("trajectory_files") or ["simple_ver"]
    max_per_domain = int(ev.get("max_samples_per_domain") or 0)
    rng = random.Random(int(ev.get("seed") or 42))

    for traj_type in traj_types:
        for domain in domains:
            root = base / traj_type / domain
            if not root.exists():
                continue
            candidates = []
            for name in traj_files:
                p = root / f"{name}.json"
                if p.exists():
                    candidates.append(p)
            for p in candidates:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    continue
                indexed = list(enumerate(data))
                if max_per_domain and len(indexed) > max_per_domain:
                    indexed = rng.sample(indexed, max_per_domain)
                for idx, sample in indexed:
                    if not sample.get("tool list"):
                        continue
                    yield {
                        "sample_id": f"{traj_type}:{domain}:{p.stem}:{idx}",
                        "domain": domain,
                        "trajectory_type": traj_type,
                        "trajectory_file": p.stem,
                        "index": idx,
                        "sample": sample,
                    }


def write_step_samples(path: Path, eval_rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            gold = row["gold_tool_list"]
            pred = row.get("pred_tool_list") or []
            pred_names = {norm_tool_name(x.get("tool name", "")) for x in pred}
            prior = []
            for step_index, tool in enumerate(gold):
                step = {
                    "sample_id": row["sample_id"],
                    "worker": row["worker"],
                    "domain": row["domain"],
                    "trajectory_type": row["trajectory_type"],
                    "step_index": step_index,
                    "query": row["query"],
                    "prior_gold_tools": prior,
                    "gold_tool": strip_outputs(tool),
                    "worker_selected_this_tool": norm_tool_name(tool.get("tool name", "")) in pred_names,
                    "trajectory_score": row["score"],
                }
                f.write(json.dumps(step, ensure_ascii=False) + "\n")
                prior.append(strip_outputs(tool))


def load_completed(path: Path, retry_failed: bool) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_failed and row.get("error"):
                continue
            sample_id = row.get("sample_id")
            worker = row.get("worker")
            if sample_id and worker:
                done.add((str(sample_id), str(worker)))
    return done


def load_rows(path: Path, retry_failed: bool) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_failed and row.get("error"):
                continue
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate OpenFugu workers on TRAJECT-Bench.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--trajectbench-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Load data/config and print planned jobs only.")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing predictions and start from scratch.")
    ap.add_argument("--retry-failed", action="store_true", help="Retry rows that previously ended with error.")
    ap.add_argument("--skip-preflight", action="store_true", help="Skip worker credential/live checks.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ev = cfg.get("evaluation") or {}
    workers = cfg.get("workers") or []
    if not workers:
        raise SystemExit("config must contain workers")

    base = Path(args.trajectbench_dir).expanduser() / "public_data"
    if not base.exists():
        raise SystemExit(f"TRAJECT-Bench public_data not found: {base}")

    samples = list(iter_samples(base, cfg))
    print(f"[trajectbench] samples={len(samples)} workers={len(workers)}", flush=True)
    if args.dry_run:
        for item in samples[:20]:
            print(f"  {item['sample_id']} query={str(item['sample'].get('query'))[:90]}")
        if not args.skip_preflight:
            preflight_workers(workers, strict=False)
        return 0

    out_cfg = cfg.get("outputs") or {}
    out_dir = Path(out_cfg.get("dir") or "results/trajectbench").expanduser()
    pred_path = Path(out_cfg.get("predictions_jsonl") or out_dir / "trajectbench_predictions.jsonl").expanduser()
    scores_path = Path(out_cfg.get("scores_csv") or out_dir / "trajectbench_scores.csv").expanduser()
    steps_path = Path(out_cfg.get("step_samples_jsonl") or out_dir / "trajectbench_step_samples.jsonl").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    temperature = float(ev.get("temperature", 0))
    max_tokens = int(ev.get("max_tokens", 4096))
    sleep_seconds = float(ev.get("sleep_seconds", 0))
    concurrency = max(1, int(ev.get("concurrency") or 1))
    request_timeout = ev.get("request_timeout")
    request_timeout = float(request_timeout) if request_timeout else None
    if not args.skip_preflight:
        try:
            preflight_workers(workers, strict=True)
        except Exception as e:
            print(str(e), file=sys.stderr, flush=True)
            return 2
    retry_failed = bool(args.retry_failed or ev.get("retry_failed"))
    resume = not args.no_resume
    completed = load_completed(pred_path, retry_failed=retry_failed) if resume else set()
    rows = load_rows(pred_path, retry_failed=retry_failed) if resume else []
    total_jobs = len(samples) * len(workers)
    ok_count = sum(1 for row in rows if not row.get("error"))
    fail_count = sum(1 for row in rows if row.get("error"))
    if completed:
        print(
            f"[trajectbench] resume enabled: loaded {len(completed)} completed worker-sample rows "
            f"from {pred_path}",
            flush=True,
        )
        print(progress_line(len(completed), total_jobs, ok_count, fail_count), flush=True)

    mode = "a" if resume else "w"
    skipped = 0
    pending = []
    for item in samples:
        tools = load_tools(base, item["domain"])
        prompt_tools = tool_catalog_for_prompt(tools)
        messages = build_prompt(
            item["sample"],
            item["domain"],
            item["trajectory_type"],
            prompt_tools,
        )
        gold = [clean_tool(x) for x in item["sample"].get("tool list", [])]
        ordered = item["trajectory_type"] == "sequential"
        for worker in workers:
            name = str(worker.get("name") or worker.get("model"))
            if (item["sample_id"], name) in completed:
                skipped += 1
                continue
            pending.append((item, worker, messages, gold, ordered))

    print(f"[trajectbench] pending jobs={len(pending)} concurrency={concurrency}", flush=True)

    with pred_path.open(mode, encoding="utf-8") as jf:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(evaluate_one, item, worker, messages, gold, ordered, temperature, max_tokens, request_timeout)
                for item, worker, messages, gold, ordered in pending
            ]
            for fut in as_completed(futures):
                row = fut.result()
                if row.get("error"):
                    print(
                        f"[trajectbench:warn] {row.get('sample_id')} {row.get('worker')} "
                        f"failed: {row.get('error')}",
                        flush=True,
                    )
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                jf.flush()
                rows.append(row)
                completed.add((item["sample_id"], name))
                if row.get("error"):
                    fail_count += 1
                else:
                    ok_count += 1
                done_count = len(completed)
                if row.get("error") or done_count == total_jobs or done_count % max(1, int(ev.get("progress_every", 10))) == 0:
                    print(progress_line(done_count, total_jobs, ok_count, fail_count, skipped), flush=True)
                if sleep_seconds and concurrency == 1:
                    time.sleep(sleep_seconds)

    with scores_path.open("w", newline="", encoding="utf-8") as cf:
        fieldnames = [
            "sample_id", "worker", "model", "domain", "trajectory_type",
            "trajectory_file", "index", "score", "name_exact", "inclusion",
            "param_accuracy", "gold_tool_count", "pred_tool_count", "latency_sec", "error_type", "error",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    write_step_samples(steps_path, rows)
    if skipped:
        print(f"[trajectbench] skipped completed rows: {skipped}", flush=True)
    print(f"[trajectbench] wrote predictions: {pred_path}", flush=True)
    print(f"[trajectbench] wrote scores: {scores_path}", flush=True)
    print(f"[trajectbench] wrote step samples: {steps_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
