#!/usr/bin/env python3
"""Evaluate OpenAI-compatible workers on BFCL's official multi-turn environment."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval_bfcl import (
    build_tools,
    call_worker,
    format_provider_error,
    load_config,
    load_jsonl,
    preflight_workers,
    provider_error_details,
)


ADDITIONAL_FUNCTION_PROMPT = (
    "I have updated some more functions you can choose from. What about now?"
)
RETRYABLE_ERRORS = {
    "auth_failed",
    "insufficient_quota",
    "rate_limited",
    "timeout",
    "parse_failed",
    "invalid_tool_schema",
    "call_failed",
}


def load_official_multiturn(bfcl_root: Path):
    package_root = bfcl_root / "berkeley-function-call-leaderboard"
    if not package_root.exists():
        raise RuntimeError(f"找不到 BFCL package: {package_root}")
    sys.path.insert(0, str(package_root))
    from bfcl_eval.constants.executable_backend_config import (  # noqa: PLC0415
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (  # noqa: PLC0415
        multi_turn_checker,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: PLC0415
        execute_multi_turn_func_call,
    )

    return (
        package_root,
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
        multi_turn_checker,
        execute_multi_turn_func_call,
    )


def bfcl_paths(package_root: Path, category: str) -> tuple[Path, Path]:
    data = package_root / "bfcl_eval" / "data"
    return (
        data / f"BFCL_v4_{category}.json",
        data / "possible_answer" / f"BFCL_v4_{category}.json",
    )


def load_function_docs(
    package_root: Path, involved_classes: list[str], mapping: dict[str, str]
) -> list[dict]:
    docs_dir = package_root / "bfcl_eval" / "data" / "multi_turn_func_doc"
    functions = []
    for class_name in involved_classes:
        filename = mapping.get(class_name)
        if not filename:
            raise RuntimeError(f"BFCL function doc mapping 缺失: {class_name}")
        functions.extend(load_jsonl(docs_dir / filename))
    return functions


def prepare_case(
    question: dict,
    category: str,
    ground_truth: list,
    package_root: Path,
    mapping: dict[str, str],
) -> dict:
    entry = copy.deepcopy(question)
    functions = load_function_docs(package_root, entry["involved_classes"], mapping)
    holdouts: dict[str, list[dict]] = {}
    for turn, names in (entry.get("missed_function") or {}).items():
        holdouts[str(turn)] = []
        for name in names:
            match = next((item for item in functions if item["name"] == name), None)
            if match is None:
                raise RuntimeError(f"{entry['id']} 找不到 holdout function: {name}")
            holdouts[str(turn)].append(match)
            functions.remove(match)
    entry["function"] = functions
    entry["missed_function"] = holdouts
    return {
        "id": str(entry["id"]),
        "category": category,
        "question": entry,
        "ground_truth": ground_truth,
    }


def iter_cases(
    package_root: Path, cfg: dict, mapping: dict[str, str]
) -> list[dict]:
    ev = cfg.get("evaluation") or {}
    categories = ev.get("categories") or ["multi_turn_base"]
    max_per_category = int(ev.get("max_samples_per_category") or 0)
    seed = int(ev.get("seed") or 42)
    cases = []
    for category_value in categories:
        category = str(category_value)
        if not category.startswith("multi_turn_"):
            raise RuntimeError(f"多轮评测不支持类别: {category}")
        question_path, answer_path = bfcl_paths(package_root, category)
        if not question_path.exists() or not answer_path.exists():
            raise RuntimeError(f"BFCL 多轮类别文件不完整: {category}")
        questions = load_jsonl(question_path)
        answers = {
            str(item["id"]): item["ground_truth"] for item in load_jsonl(answer_path)
        }
        if max_per_category and len(questions) > max_per_category:
            rng = random.Random(f"{seed}:{category}")
            questions = rng.sample(questions, max_per_category)
        for question in questions:
            case_id = str(question["id"])
            if case_id not in answers:
                raise RuntimeError(f"BFCL ground truth 缺失: {case_id}")
            cases.append(
                prepare_case(question, category, answers[case_id], package_root, mapping)
            )
    return cases


def function_calls(response) -> tuple[list[dict], list[str], dict, str]:
    message = response.choices[0].message
    content = message.content or ""
    parsed = []
    executable = []
    wire_calls = []
    for index, tool_call in enumerate(message.tool_calls or []):
        function = tool_call.get("function") if isinstance(tool_call, dict) else tool_call.function
        name = function.get("name") if isinstance(function, dict) else function.name
        arguments = function.get("arguments") if isinstance(function, dict) else function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments 不是 JSON object")
        call_id = (
            tool_call.get("id")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "id", None)
        ) or f"call_{index}"
        name = str(name)
        parsed.append({name: arguments})
        executable.append(
            f"{name}({','.join(f'{key}={value!r}' for key, value in arguments.items())})"
        )
        wire_calls.append({
            "id": str(call_id),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
    assistant = {"role": "assistant", "content": content or None}
    if wire_calls:
        assistant["tool_calls"] = wire_calls
    return parsed, executable, assistant, content


def evaluate_one(
    case: dict,
    worker: dict,
    ev: dict,
    multi_turn_checker,
    execute_multi_turn_func_call,
    run_nonce: str,
) -> dict:
    worker_name = str(worker.get("name") or worker["model"])
    entry = copy.deepcopy(case["question"])
    functions = list(entry["function"])
    holdouts = entry.get("missed_function") or {}
    messages: list[dict] = []
    decoded_turns: list[list[list[str]]] = []
    trajectory = []
    api_calls = 0
    max_steps = max(1, int(ev.get("max_steps_per_turn") or 20))
    execution_model = f"openfugu_{worker_name}_{run_nonce}"
    started = time.time()
    row = {
        "case_id": case["id"],
        "category": case["category"],
        "worker": worker_name,
        "model": worker["model"],
        "question": entry["question"],
        "ground_truth": case["ground_truth"],
    }

    try:
        for turn_index, original_turn_messages in enumerate(entry["question"]):
            turn_messages = [dict(item) for item in original_turn_messages]
            if str(turn_index) in holdouts:
                functions.extend(holdouts[str(turn_index)])
                if turn_messages:
                    raise RuntimeError(f"{case['id']} holdout turn 应为空")
                turn_messages = [
                    {"role": "user", "content": ADDITIONAL_FUNCTION_PROMPT}
                ]
            messages.extend(turn_messages)
            turn_steps = []
            for step_index in range(max_steps + 1):
                if step_index == max_steps:
                    raise RuntimeError(f"超过每轮最大模型调用步数 {max_steps}")
                context_before = copy.deepcopy(messages)
                response = call_worker(worker, messages, build_tools(functions), ev)
                api_calls += 1
                parsed, executable, assistant, content = function_calls(response)
                messages.append(assistant)
                step = {
                    "turn": turn_index,
                    "step": step_index,
                    "context": context_before,
                    "available_functions": copy.deepcopy(functions),
                    "prediction": parsed,
                    "prediction_executable": executable,
                    "content": content,
                    "execution_results": [],
                }
                trajectory.append(step)
                if not executable:
                    break
                turn_steps.append(executable)
                results, _ = execute_multi_turn_func_call(
                    executable,
                    entry.get("initial_config") or {},
                    entry["involved_classes"],
                    execution_model,
                    case["id"],
                    long_context="long_context" in case["category"],
                    is_evaL_run=False,
                )
                step["execution_results"] = list(results)
                for tool_call, result in zip(assistant["tool_calls"], results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result),
                    })
            decoded_turns.append(turn_steps)

        checker_result = multi_turn_checker(
            decoded_turns,
            case["ground_truth"],
            copy.deepcopy(entry),
            case["category"],
            execution_model,
        )
        valid = bool(checker_result["valid"])
        row.update({
            "prediction_decoded": decoded_turns,
            "trajectory": trajectory,
            "valid": valid,
            "score": float(valid),
            "api_calls": api_calls,
            "turns": len(decoded_turns),
        })
        if not valid:
            row["error_type"] = checker_result.get("error_type")
            row["error"] = checker_result.get("error_message")
            row["error_details"] = checker_result.get("details")
    except Exception as error:
        details = provider_error_details(error, worker)
        error_type = details["error_type"]
        if "超过每轮最大模型调用步数" in str(error):
            error_type = "multi_turn:force_terminated"
        row.update({
            "prediction_decoded": decoded_turns,
            "trajectory": trajectory,
            "valid": False,
            "score": 0.0,
            "api_calls": api_calls,
            "turns": len(decoded_turns),
            "error_type": error_type,
            "provider": details["provider"],
            "provider_code": details["provider_code"],
            "provider_message": details["provider_message"],
            "resource_hint": details["resource_hint"],
            "error": details["raw_error"],
        })
    row["latency_sec"] = round(time.time() - started, 3)
    return row


def load_existing(
    path: Path, retry_failed: bool, allowed_keys: set[tuple[str, str]]
) -> tuple[list[dict], set[tuple[str, str]]]:
    latest = {}
    if path.exists():
        for row in load_jsonl(path):
            key = (str(row.get("case_id")), str(row.get("worker")))
            if key in allowed_keys:
                latest[key] = row
    if retry_failed:
        latest = {
            key: row
            for key, row in latest.items()
            if row.get("error_type") not in RETRYABLE_ERRORS
        }
    return list(latest.values()), set(latest)


def write_scores(path: Path, rows: list[dict]) -> None:
    fields = [
        "case_id",
        "category",
        "worker",
        "model",
        "score",
        "valid",
        "turns",
        "api_calls",
        "latency_sec",
        "error_type",
        "provider",
        "provider_code",
        "provider_message",
        "resource_hint",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_step_samples(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for step in row.get("trajectory") or []:
                sample = {
                    "case_id": row["case_id"],
                    "category": row["category"],
                    "worker": row["worker"],
                    "model": row["model"],
                    "episode_valid": row.get("valid", False),
                    **step,
                }
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 BFCL 官方环境评测多轮工具调用。")
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
    package_root, mapping, checker, executor = load_official_multiturn(
        Path(args.bfcl_dir).expanduser()
    )
    cases = iter_cases(package_root, cfg, mapping)
    ev = cfg.get("evaluation") or {}
    total = len(cases) * len(workers)
    turns = sum(len(case["question"]["question"]) for case in cases)
    print("[bfcl-mt] 评分器=BFCL 官方 multi_turn_checker", flush=True)
    print("[bfcl-mt] 工具环境=BFCL 官方本地可执行 API", flush=True)
    print(
        f"[bfcl-mt] episode={len(cases)} 用户轮次={turns} worker={len(workers)} "
        f"episode-worker 总数={total}",
        flush=True,
    )
    if args.dry_run:
        for case in cases[:20]:
            print(
                f"  {case['id']} [{case['category']}] "
                f"turns={len(case['question']['question'])} "
                f"tools={len(case['question']['function'])}",
                flush=True,
            )
        return 0

    if not args.skip_preflight:
        try:
            preflight_workers(workers, ev)
        except Exception as error:
            print(f"[bfcl-mt:preflight:error] {error}", file=sys.stderr, flush=True)
            return 2

    outputs = cfg.get("outputs") or {}
    out_dir = Path(outputs.get("dir") or "openfugu_bfcl_multiturn").expanduser()
    pred_path = Path(
        outputs.get("predictions_jsonl")
        or out_dir / "bfcl_multiturn_predictions.jsonl"
    ).expanduser()
    score_path = Path(
        outputs.get("scores_csv") or out_dir / "bfcl_multiturn_scores.csv"
    ).expanduser()
    steps_path = Path(
        outputs.get("step_samples_jsonl")
        or out_dir / "bfcl_multiturn_step_samples.jsonl"
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_keys = {
        (case["id"], str(worker.get("name") or worker["model"]))
        for case in cases
        for worker in workers
    }
    resume = not args.no_resume
    rows, completed = (
        load_existing(pred_path, args.retry_failed, allowed_keys)
        if resume
        else ([], set())
    )
    pending = [
        (case, worker)
        for case in cases
        for worker in workers
        if (case["id"], str(worker.get("name") or worker["model"])) not in completed
    ]
    concurrency = max(1, int(ev.get("concurrency") or 1))
    progress_every = max(1, int(ev.get("progress_every") or 1))
    print(
        f"[bfcl-mt] 已完成={len(completed)} 待执行={len(pending)} 并发={concurrency}",
        flush=True,
    )
    mode = "a" if resume else "w"
    run_nonce = uuid.uuid4().hex[:10]
    with pred_path.open(mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(evaluate_one, case, worker, ev, checker, executor, run_nonce)
                for case, worker in pending
            ]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed.add((row["case_id"], row["worker"]))
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                done = len(completed)
                if row.get("error_type") or done == total or done % progress_every == 0:
                    if row.get("provider_message"):
                        worker = next(
                            item
                            for item in workers
                            if str(item.get("name") or item["model"]) == row["worker"]
                        )
                        details = provider_error_details(RuntimeError(row["error"]), worker)
                        details.update({
                            "error_type": row.get("error_type", ""),
                            "provider_code": row.get("provider_code", ""),
                            "provider_message": row.get("provider_message", ""),
                            "resource_hint": row.get("resource_hint", ""),
                        })
                        print(
                            "[bfcl-mt:worker:error] " + format_provider_error(details),
                            flush=True,
                        )
                    passed = sum(bool(item.get("valid")) for item in rows)
                    calls = sum(int(item.get("api_calls") or 0) for item in rows)
                    print(
                        f"[bfcl-mt] 进度 {done}/{total} ({100 * done / max(1, total):.1f}%) "
                        f"通过={passed} 未通过={len(rows) - passed} 模型调用={calls}",
                        flush=True,
                    )

    latest = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["worker"]))
        if key in allowed_keys:
            latest[key] = row
    final_rows = list(latest.values())
    write_scores(score_path, final_rows)
    write_step_samples(steps_path, final_rows)
    print(f"[bfcl-mt] 已写出预测: {pred_path}", flush=True)
    print(f"[bfcl-mt] 已写出评分: {score_path}", flush=True)
    print(f"[bfcl-mt] 已写出逐步样本: {steps_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
