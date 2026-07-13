#!/usr/bin/env python3
"""Evaluate arbitrary OpenAI-compatible workers with BFCL V4's AST checker."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import re
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CHECKER_MODEL_NAME = "OpenFugu-FC"


def load_config(path: str) -> dict:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("YAML 配置需要 PyYAML。") from e
    return yaml.safe_load(text) or {}


def resolve_secret(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("env:"):
        name = value.split(":", 1)[1]
        result = os.environ.get(name)
        if not result:
            raise RuntimeError(f"环境变量 {name} 未设置")
        return result
    return value


def load_official_checker(bfcl_root: Path):
    package_root = bfcl_root / "berkeley-function-call-leaderboard"
    checker_file = package_root / "bfcl_eval" / "eval_checker" / "ast_eval" / "ast_checker.py"
    if not checker_file.exists():
        raise RuntimeError(f"找不到 BFCL 官方 AST checker: {checker_file}")

    # ast_checker only needs underscore_to_dot from model_config. Stubbing this
    # avoids importing every BFCL provider and its unrelated heavy dependencies.
    stub = types.ModuleType("bfcl_eval.constants.model_config")
    stub.MODEL_CONFIG_MAPPING = {
        CHECKER_MODEL_NAME: SimpleNamespace(underscore_to_dot=True)
    }
    sys.modules["bfcl_eval.constants.model_config"] = stub
    sys.path.insert(0, str(package_root))
    try:
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    finally:
        sys.path.remove(str(package_root))
    return ast_checker, Language


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bfcl_paths(bfcl_root: Path, category: str) -> tuple[Path, Path]:
    data = bfcl_root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    return (
        data / f"BFCL_v4_{category}.json",
        data / "possible_answer" / f"BFCL_v4_{category}.json",
    )


def iter_cases(bfcl_root: Path, cfg: dict) -> list[dict]:
    ev = cfg.get("evaluation") or {}
    categories = ev.get("categories") or ["simple_python"]
    max_per_category = int(ev.get("max_samples_per_category") or 0)
    seed = int(ev.get("seed") or 42)
    cases = []
    for category in categories:
        question_path, answer_path = bfcl_paths(bfcl_root, str(category))
        if not question_path.exists() or not answer_path.exists():
            raise RuntimeError(f"BFCL 类别文件不完整: {category}")
        questions = load_jsonl(question_path)
        answers = {x["id"]: x["ground_truth"] for x in load_jsonl(answer_path)}
        if max_per_category and len(questions) > max_per_category:
            rng = random.Random(f"{seed}:{category}")
            questions = rng.sample(questions, max_per_category)
        for question in questions:
            case_id = str(question["id"])
            if case_id not in answers:
                raise RuntimeError(f"BFCL ground truth 缺失: {case_id}")
            cases.append({
                "id": case_id,
                "category": str(category),
                "question": question,
                "ground_truth": answers[case_id],
            })
    return cases


def sanitize_function_name(name: str) -> str:
    return name.replace(".", "_")


def normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_schema(x) for x in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key == "optional":
            continue
        out[key] = normalize_schema(item)
    type_name = out.get("type")
    if type_name == "dict":
        out["type"] = "object"
    elif type_name == "tuple":
        out["type"] = "array"
    elif type_name == "any":
        out.pop("type", None)
    return out


def build_tools(functions: list[dict]) -> list[dict]:
    tools = []
    for function in functions:
        tools.append({
            "type": "function",
            "function": {
                "name": sanitize_function_name(str(function["name"])),
                "description": str(function.get("description") or ""),
                "parameters": normalize_schema(copy.deepcopy(function.get("parameters") or {})),
            },
        })
    return tools


def case_messages(question: dict) -> list[dict]:
    turns = question.get("question") or []
    if not turns or not isinstance(turns[0], list):
        raise ValueError("BFCL single-turn question 格式不正确")
    return [dict(message) for message in turns[0]]


def call_worker(worker: dict, messages: list[dict], tools: list[dict], ev: dict):
    import litellm

    kwargs = {
        "model": worker["model"],
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": float(ev.get("temperature", 0)),
        "max_tokens": int(ev.get("max_tokens", 2048)),
    }
    timeout = ev.get("request_timeout")
    if timeout:
        kwargs["timeout"] = float(timeout)
    if worker.get("api_base") or worker.get("base_url"):
        kwargs["api_base"] = resolve_secret(worker.get("api_base") or worker.get("base_url"))
    if worker.get("api_key"):
        kwargs["api_key"] = resolve_secret(worker["api_key"])
    kwargs.update(worker.get("extra_params") or {})
    return litellm.completion(**kwargs)


def parse_tool_calls(response) -> tuple[list[dict], str]:
    message = response.choices[0].message
    content = message.content or ""
    parsed = []
    for tool_call in message.tool_calls or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else tool_call.function
        name = function.get("name") if isinstance(function, dict) else function.name
        arguments = function.get("arguments") if isinstance(function, dict) else function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments 不是 JSON object")
        parsed.append({str(name): arguments})
    return parsed, content


def classify_error(error: Exception) -> str:
    text = str(error).lower()
    if any(token in text for token in ("余额不足", "无可用资源包", "insufficient_quota", "insufficient funds", "credit balance")):
        return "insufficient_quota"
    if "authentication" in text or "身份验证" in text:
        return "auth_failed"
    if "rate limit" in text or "429" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if isinstance(error, json.JSONDecodeError):
        return "parse_failed"
    return "call_failed"


def provider_error_details(error: Exception, worker: dict) -> dict:
    raw = str(error)
    payloads = [raw]
    body = getattr(error, "body", None)
    if body:
        payloads.append(json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body)
    response = getattr(error, "response", None)
    if response is not None:
        try:
            payloads.append(json.dumps(response.json(), ensure_ascii=False))
        except Exception:
            response_text = getattr(response, "text", "")
            if response_text:
                payloads.append(str(response_text))
    payload_text = "\n".join(payloads)
    code_matches = re.findall(r"['\"]?code['\"]?\s*:\s*['\"]?([^'\"},\s]+)", payload_text)
    message_matches = re.findall(r"['\"]?message['\"]?\s*:\s*['\"]([^'\"]+)", payload_text)
    endpoint = str(worker.get("api_base") or worker.get("base_url") or "<default>")
    model = str(worker.get("model") or "<unknown>")
    worker_name = str(worker.get("name") or model)
    error_type = classify_error(RuntimeError(payload_text))

    if "bigmodel.cn" in endpoint:
        provider = "智谱开放平台"
    elif "deepseek.com" in endpoint:
        provider = "DeepSeek 官方平台"
    else:
        provider = endpoint

    if error_type == "insufficient_quota":
        resource_hint = (
            f"{provider} 的账户余额或模型 {model.removeprefix('openai/')} 可用资源包；"
            "供应商响应未给出具体资源包名称，请到对应控制台查看套餐/余额明细"
        )
    elif error_type == "rate_limited":
        resource_hint = f"{provider} 对模型 {model.removeprefix('openai/')} 的并发或速率配额"
    else:
        resource_hint = ""

    return {
        "error_type": error_type,
        "provider": provider,
        "provider_code": code_matches[-1] if code_matches else "",
        "provider_message": message_matches[-1] if message_matches else raw,
        "resource_hint": resource_hint,
        "worker": worker_name,
        "model": model,
        "endpoint": endpoint,
        "raw_error": raw,
    }


def format_provider_error(details: dict) -> str:
    parts = [
        f"worker={details['worker']}",
        f"model={details['model']}",
        f"endpoint={details['endpoint']}",
        f"类型={details['error_type']}",
    ]
    if details.get("provider_code"):
        parts.append(f"供应商错误码={details['provider_code']}")
    if details.get("provider_message"):
        parts.append(f"供应商消息={details['provider_message']}")
    if details.get("resource_hint"):
        parts.append(f"需检查={details['resource_hint']}")
    return " | ".join(parts)


def evaluate_one(case: dict, worker: dict, ev: dict, ast_checker, Language) -> dict:
    worker_name = str(worker.get("name") or worker["model"])
    question = case["question"]
    row = {
        "case_id": case["id"],
        "category": case["category"],
        "worker": worker_name,
        "model": worker["model"],
        "question": question["question"],
        "function": question["function"],
        "ground_truth": case["ground_truth"],
    }
    started = time.time()
    try:
        response = call_worker(worker, case_messages(question), build_tools(question["function"]), ev)
        prediction, content = parse_tool_calls(response)
        row["prediction"] = prediction
        row["content"] = content
        result = ast_checker(
            question["function"], prediction, case["ground_truth"],
            Language.PYTHON, case["category"], CHECKER_MODEL_NAME,
        )
        row["valid"] = bool(result["valid"])
        row["score"] = float(row["valid"])
        if not row["valid"]:
            row["error_type"] = result.get("error_type")
            row["error"] = result.get("error")
    except Exception as e:
        details = provider_error_details(e, worker)
        row.update({
            "prediction": [],
            "valid": False,
            "score": 0.0,
            "error_type": details["error_type"],
            "provider": details["provider"],
            "provider_code": details["provider_code"],
            "provider_message": details["provider_message"],
            "resource_hint": details["resource_hint"],
            "error": details["raw_error"],
        })
    row["latency_sec"] = round(time.time() - started, 3)
    return row


def load_existing(path: Path, retry_failed: bool) -> tuple[list[dict], set[tuple[str, str]]]:
    latest = {}
    if path.exists():
        for row in load_jsonl(path):
            latest[(str(row.get("case_id")), str(row.get("worker")))] = row
    if retry_failed:
        retryable = {"auth_failed", "rate_limited", "timeout", "parse_failed", "call_failed"}
        latest = {key: row for key, row in latest.items() if row.get("error_type") not in retryable}
    return list(latest.values()), set(latest)


def write_scores(path: Path, rows: list[dict]) -> None:
    fields = [
        "case_id", "category", "worker", "model", "score", "valid", "latency_sec",
        "error_type", "provider", "provider_code", "provider_message", "resource_hint", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def preflight_workers(workers: list[dict], ev: dict) -> None:
    print("[bfcl] 预检 worker 原生 tool calling", flush=True)
    messages = [{"role": "user", "content": "Call ping with value 1."}]
    tools = [{"type": "function", "function": {"name": "ping", "description": "Ping", "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
    check_ev = dict(ev)
    check_ev["max_tokens"] = 64
    check_ev["request_timeout"] = min(float(ev.get("request_timeout") or 30), 30)
    for worker in workers:
        name = str(worker.get("name") or worker["model"])
        try:
            response = call_worker(worker, messages, tools, check_ev)
            calls, _ = parse_tool_calls(response)
            if not calls or calls[0].get("ping", {}).get("value") != 1:
                raise RuntimeError(f"未返回有效原生 tool call: {calls}")
            print(f"[bfcl]   {name}: 通过", flush=True)
        except Exception as e:
            raise RuntimeError(format_provider_error(provider_error_details(e, worker))) from None


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 BFCL V4 官方 AST checker 评测 OpenFugu workers。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bfcl-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    workers = cfg.get("workers") or []
    if not workers:
        raise SystemExit("配置中没有 workers")
    bfcl_root = Path(args.bfcl_dir).expanduser()
    ast_checker, Language = load_official_checker(bfcl_root)
    cases = iter_cases(bfcl_root, cfg)
    ev = cfg.get("evaluation") or {}
    print("[bfcl] 评分器=BFCL V4 官方 AST checker", flush=True)
    print(f"[bfcl] 用例数={len(cases)} worker 数={len(workers)} 调用总数={len(cases) * len(workers)}", flush=True)
    if args.dry_run:
        for case in cases[:20]:
            content = case["question"]["question"][0][0].get("content", "")
            print(f"  {case['id']} [{case['category']}] {content[:100]}")
        return 0

    if not args.skip_preflight:
        try:
            preflight_workers(workers, ev)
        except Exception as e:
            print(f"[bfcl:preflight:error] {e}", file=sys.stderr, flush=True)
            return 2

    outputs = cfg.get("outputs") or {}
    out_dir = Path(outputs.get("dir") or "openfugu_bfcl").expanduser()
    pred_path = Path(outputs.get("predictions_jsonl") or out_dir / "bfcl_predictions.jsonl").expanduser()
    score_path = Path(outputs.get("scores_csv") or out_dir / "bfcl_scores.csv").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    resume = not args.no_resume
    rows, completed = load_existing(pred_path, args.retry_failed) if resume else ([], set())
    pending = []
    for case in cases:
        for worker in workers:
            key = (case["id"], str(worker.get("name") or worker["model"]))
            if key not in completed:
                pending.append((case, worker))

    total = len(cases) * len(workers)
    print(f"[bfcl] 已完成={len(completed)} 待执行={len(pending)} 并发={int(ev.get('concurrency') or 1)}", flush=True)
    mode = "a" if resume else "w"
    concurrency = max(1, int(ev.get("concurrency") or 1))
    progress_every = max(1, int(ev.get("progress_every") or 10))
    with pred_path.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(evaluate_one, case, worker, ev, ast_checker, Language) for case, worker in pending]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed.add((row["case_id"], row["worker"]))
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                done = len(completed)
                if row.get("error_type") or done == total or done % progress_every == 0:
                    ok = sum(bool(x.get("valid")) for x in rows)
                    if row.get("provider_message"):
                        print("[bfcl:worker:error] " + format_provider_error({
                            "worker": row["worker"],
                            "model": row["model"],
                            "endpoint": next(
                                str(w.get("api_base") or w.get("base_url") or "<default>")
                                for w in workers if str(w.get("name") or w["model"]) == row["worker"]
                            ),
                            "error_type": row.get("error_type", ""),
                            "provider_code": row.get("provider_code", ""),
                            "provider_message": row.get("provider_message", ""),
                            "resource_hint": row.get("resource_hint", ""),
                        }), flush=True)
                    print(f"[bfcl] 进度 {done}/{total} ({100 * done / max(1, total):.1f}%) 通过={ok} 未通过={len(rows) - ok}", flush=True)

    latest = {}
    for row in rows:
        latest[(row["case_id"], row["worker"])] = row
    final_rows = list(latest.values())
    write_scores(score_path, final_rows)
    print(f"[bfcl] 已写出预测: {pred_path}", flush=True)
    print(f"[bfcl] 已写出评分: {score_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
