#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: web console for the unified evaluator (EvalScope). Original code.
"""
eval_ui.py — 统一评估器（EvalScope）的 Web 控制台。

命令行三步（evalscope eval × N worker → evalscope_to_matrix → 训练）在这里变成
点按钮：选 worker / 基准 / 条数 → 发起评测 → 实时进度与日志 → 一键生成训练矩阵。

stdlib http.server only — 与 serve.py 同款，不引入 web 框架。评测本身跑在
.venv-eval 隔离环境的 evalscope 子进程里，本服务只做编排与展示。

Run:
  export DEEPSEEK_API_KEY=... ZHIPU_API_KEY=...
  python openfugu/eval_ui.py --config configs/bfcl.yaml --port 8090
  # 浏览器打开 http://127.0.0.1:8090
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from evalscope_to_matrix import classify, find_reviews_dir, load_run  # 复用同一套失败分类，口径一致

EVALSCOPE = Path(os.environ.get("FUGU_EVALSCOPE", ROOT / ".venv-eval/bin/evalscope"))
HTML_PATH = Path(__file__).with_name("eval_ui.html")
WORK_ROOT = ROOT / "openfugu_evalscope"
OUT_DIR = ROOT / "openfugu_bfcl"
BFCL_DATA = ROOT / "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
MS_CACHE = Path.home() / ".cache/modelscope/hub/datasets"

# 界面下拉里给的常用基准；也允许前端手填任意 EvalScope 基准名。
BENCHMARKS = [
    {"name": "bfcl_v4", "label": "BFCL V4 工具调用", "subsets":
        ["simple_python", "parallel", "multiple", "parallel_multiple"]},
    {"name": "gsm8k", "label": "GSM8K 数学应用题", "subsets": []},
    {"name": "math_500", "label": "MATH-500 竞赛数学", "subsets": []},
    {"name": "aime25", "label": "AIME 2025", "subsets": []},
    {"name": "ifeval", "label": "IFEval 指令遵循", "subsets": []},
    {"name": "gaia", "label": "GAIA 多轮 agent", "subsets": []},
    {"name": "acebench", "label": "AceBench 工具 agent", "subsets": []},
]

CONFIG: dict = {}
CFG_PATH: Path = ROOT / "configs/bfcl.yaml"
WORKERS: list[dict] = []          # [{name, model, raw_model, api_base, key_env}]
JOBS: dict[str, dict] = {}        # worker name -> job 状态
DOWNLOADS: dict[str, dict] = {}   # dataset -> 数据集下载状态 {status, error}
LOCK = threading.Lock()
NAME_RE = re.compile(r"^[\w.\- ]{1,64}$")


def load_config(path: Path) -> None:
    global CONFIG, CFG_PATH, WORKERS
    CFG_PATH = path.resolve()
    CONFIG = yaml.safe_load(path.read_text(encoding="utf-8"))
    WORKERS = []
    for w in CONFIG.get("workers", []):
        model = w["model"]
        # litellm 形如 openai/deepseek-v4-flash；evalscope 只要裸模型名
        if model.startswith("openai/"):
            model = model.split("/", 1)[1]
        key = str(w.get("api_key", ""))
        WORKERS.append({
            "name": w["name"], "model": model, "raw_model": w["model"],
            "api_base": w["api_base"],
            "key_env": key[4:] if key.startswith("env:") else "",
        })


def save_workers() -> tuple[Path, str]:
    """把 WORKERS 写回配置的 workers: 段，其余段落与注释原样保留。

    回退配置（example）只读不写：首次保存时以它为底版生成 configs/bfcl.yaml。
    """
    target = CFG_PATH if CFG_PATH.name != "bfcl.example.yaml" else ROOT / "configs/bfcl.yaml"
    src = target if target.exists() else CFG_PATH
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    block = ["workers:\n"]
    for w in WORKERS:
        block += [f"  - name: {w['name']}\n",
                  f"    model: {w['raw_model']}\n",
                  f"    api_base: {w['api_base']}\n"]
        if w["key_env"]:
            block.append(f"    api_key: env:{w['key_env']}\n")
        block.append("\n")
    try:
        start = next(i for i, l in enumerate(lines) if re.match(r"^workers:\s*$", l))
    except StopIteration:
        lines.append("\n")
        start = len(lines)
    end = start + 1
    while end < len(lines) and not re.match(r"^[A-Za-z_]\w*:", lines[end]):
        end += 1
    text = "".join(lines[:start] + block + lines[end:])
    target.write_text(text.rstrip("\n") + "\n", encoding="utf-8")  # 尾部空行不累积
    return target, str(target.relative_to(ROOT))


def edit_workers(body: dict) -> tuple[int, dict]:
    global WORKERS
    action = body.get("action")
    if action == "delete":
        before = len(WORKERS)
        WORKERS = [w for w in WORKERS if w["name"] != body.get("name")]
        if len(WORKERS) == before:
            return 404, {"error": "worker 不存在"}
    elif action in ("add", "update"):
        w = body.get("worker") or {}
        name = (w.get("name") or "").strip()
        model = (w.get("model") or "").strip()
        api_base = (w.get("api_base") or "").strip().rstrip("/")
        key_env = (w.get("key_env") or "").strip()
        if not NAME_RE.match(name):
            return 400, {"error": "名称只允许字母数字空格、. - _"}
        if not re.fullmatch(r"[\w.\-/]+", model or ""):
            return 400, {"error": "模型 ID 不合法"}
        if not api_base.startswith("http"):
            return 400, {"error": "API 地址需要 http(s) 开头"}
        if key_env and not re.fullmatch(r"[A-Z][A-Z0-9_]*", key_env):
            return 400, {"error": "key 环境变量名需要大写字母/数字/下划线"}
        # 写回 YAML 时保持 litellm 风格（prepare_bfcl.sh 也吃这个配置）
        raw = model if "/" in model else f"openai/{model}"
        entry = {"name": name,
                 "model": raw.split("/", 1)[1] if raw.startswith("openai/") else raw,
                 "raw_model": raw, "api_base": api_base, "key_env": key_env}
        idx = next((i for i, x in enumerate(WORKERS)
                    if x["name"] == body.get("orig_name")), None)
        if idx is None:
            if any(x["name"] == name for x in WORKERS):
                return 400, {"error": f"同名 worker 已存在: {name}"}
            WORKERS.append(entry)
        else:
            WORKERS[idx] = entry
    else:
        return 400, {"error": "action 需要 add/update/delete"}
    _, saved = save_workers()
    return 200, {"ok": True, "saved_to": saved}


def inject_key(body: dict) -> tuple[int, dict]:
    """把 key 注入本进程环境变量，不落盘；重启服务后失效。"""
    env = (body.get("env") or "").strip()
    value = (body.get("value") or "").strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env) or not value:
        return 400, {"error": "需要合法的环境变量名和非空 value"}
    os.environ[env] = value
    return 200, {"ok": True}


def test_worker(body: dict) -> tuple[int, dict]:
    """发一条 1 token 的真实请求验证 key/端点/模型名都对。"""
    w = next((x for x in WORKERS if x["name"] == body.get("name")), None)
    if not w:
        return 404, {"error": "worker 不存在"}
    key = os.environ.get(w["key_env"], "") if w["key_env"] else ""
    req = urllib.request.Request(
        w["api_base"].rstrip("/") + "/chat/completions",
        data=json.dumps({"model": w["model"], "max_tokens": 1,
                         "messages": [{"role": "user", "content": "ping"}]}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            json.loads(resp.read())
        return 200, {"ok": True, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return 200, {"ok": False, "error": str(e)[:200]}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def tail_progress(log_path: Path) -> tuple[int, list[str]]:
    """从日志尾部提取 tqdm 百分比与最近几行（去掉 \r 覆盖段）。"""
    try:
        raw = log_path.read_bytes()[-8192:].decode("utf-8", "replace")
    except OSError:
        return 0, []
    lines = [seg for line in raw.splitlines() for seg in [line.split("\r")[-1]] if seg.strip()]
    pct = 0
    for m in re.finditer(r"(\d+)%\|", raw):
        pct = int(m.group(1))
    return pct, lines[-8:]


def summarize(worker: str, workdir: Path) -> dict:
    """评测跑完后按转换器同一口径统计：通过 / 判 0 / 执行失败。"""
    rows = load_run(worker, workdir, metric=None)
    n = len(rows)
    transport = [r for r in rows.values() if r["status"] == "transport_error"]
    passed = sum(1 for r in rows.values() if r["status"] == "scored" and r["score"] > 0)
    return {"total": n, "passed": passed, "transport_errors": len(transport),
            "errors": [f"{r['case_id']}: {r['error'][:100]}" for r in transport[:5]]}


def run_job(worker: dict, dataset: str, subsets: list[str], limit: int, timeout: int,
            use_cache: Path | None = None):
    name = worker["name"]
    workdir = WORK_ROOT / slug(name)
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "eval_ui.log"
    cmd = [str(EVALSCOPE), "eval",
           "--model", worker["model"],
           "--api-url", worker["api_base"],
           "--api-key", os.environ.get(worker["key_env"], ""),
           "--eval-type", "openai_api",
           "--datasets", dataset,
           "--generation-config", f"timeout={timeout}",
           "--work-dir", str(workdir)]
    if use_cache:
        # 续跑模式：复用时间戳目录里未被剪掉的缓存，只补跑缺失样本
        cmd += ["--use-cache", str(use_cache)]
    if subsets:
        cmd += ["--dataset-args", json.dumps({dataset: {"subset_list": subsets}})]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    # 记住本次参数，重试时原样复用（换参数会破坏样本对齐）
    (workdir / "last_run.json").write_text(json.dumps(
        {"dataset": dataset, "subsets": subsets, "limit": limit, "timeout": timeout}))
    with LOCK:
        JOBS[name] = {"worker": name, "status": "running", "workdir": str(workdir),
                      "log": str(log_path), "started": time.time(), "summary": None,
                      "error": "", "retried": bool(use_cache)}
    with log_path.open("w", encoding="utf-8") as log:
        # 日志首行回显命令便于排查，但把 key 打码
        shown = [("***" if worker["key_env"] and a == os.environ.get(worker["key_env"]) else a)
                 for a in cmd]
        log.write("$ " + " ".join(shown) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    with LOCK:
        job = JOBS[name]
        job["finished"] = time.time()
        if rc != 0:
            job["status"], job["error"] = "failed", f"evalscope 退出码 {rc}，见日志"
            return
        try:
            job["summary"] = summarize(name, Path(workdir))
            job["status"] = "done"
        except SystemExit as e:
            job["status"], job["error"] = "failed", str(e)


def prune_failed_cache(workdir: Path) -> tuple[Path, int]:
    """把执行失败的样本从 predictions/reviews 两级缓存里删掉。

    EvalScope 的 --use-cache 不分成败一律复用，剪掉失败行后它才会只补跑
    这些样本。返回 (时间戳目录, 剪掉的样本数)。
    """
    ts_dir = find_reviews_dir(workdir).parent
    pruned = 0
    for rev_file in sorted((ts_dir / "reviews").glob("*/*.jsonl")):
        entries = [json.loads(l) for l in rev_file.open(encoding="utf-8") if l.strip()]
        bad = {e["index"] for e in entries
               if classify(e["sample_score"]["score"])[0] == "transport_error"}
        if not bad:
            continue
        pruned += len(bad)
        keep = [e for e in entries if e["index"] not in bad]
        with rev_file.open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in keep)
        pred_file = ts_dir / "predictions" / rev_file.parent.name / rev_file.name
        if pred_file.exists():
            preds = [json.loads(l) for l in pred_file.open(encoding="utf-8") if l.strip()]
            with pred_file.open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(p, ensure_ascii=False) + "\n"
                             for p in preds if p["index"] not in bad)
    return ts_dir, pruned


def retry_failed(body: dict) -> tuple[int, dict]:
    name = body.get("worker") or ""
    worker = next((w for w in WORKERS if w["name"] == name), None)
    if not worker:
        return 404, {"error": f"worker 不存在: {name}"}
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            return 409, {"error": "已有评测在跑，等它结束"}
    workdir = WORK_ROOT / slug(name)
    params_file = workdir / "last_run.json"
    if not params_file.exists():
        return 400, {"error": "找不到上次运行参数，请重新发起评测"}
    params = json.loads(params_file.read_text())
    ts_dir, pruned = prune_failed_cache(workdir)
    if pruned == 0:
        return 400, {"error": "没有执行失败的条目，无需重试"}
    threading.Thread(target=run_job,
                     args=(worker, params["dataset"], params["subsets"],
                           params["limit"], params["timeout"], ts_dir),
                     daemon=True).start()
    return 200, {"retrying": pruned, "cache": str(ts_dir)}


def start_runs(body: dict) -> tuple[int, dict]:
    if not EVALSCOPE.exists():
        return 400, {"error": f"找不到 {EVALSCOPE}，先运行 bash scripts/setup_eval_env.sh"}
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            return 409, {"error": "已有评测在跑，等它结束或重启服务"}
    names = body.get("workers") or []
    picked = [w for w in WORKERS if w["name"] in names]
    if not picked:
        return 400, {"error": "至少选一个 worker"}
    missing = [w["name"] for w in picked if w["key_env"] and not os.environ.get(w["key_env"])]
    if missing:
        return 400, {"error": f"缺少 API key: {missing}，在模型配置里注入或 export 后重启"}
    dataset = (body.get("dataset") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", dataset or ""):
        return 400, {"error": f"基准名不合法: {dataset!r}"}
    subsets = [s.strip() for s in body.get("subsets", []) if s.strip()]
    limit = int(body.get("limit", 0))
    timeout = int(body.get("timeout", 600))
    JOBS.clear()
    for w in picked:
        threading.Thread(target=run_job, args=(w, dataset, subsets, limit, timeout),
                         daemon=True).start()
    return 200, {"started": [w["name"] for w in picked]}


def _preview_bfcl(subsets: list[str], n: int) -> tuple[int, dict]:
    """BFCL 题库就在仓库里（JSONL），按子集分组取前 n 条。"""
    groups = []
    for sub in subsets:
        f = BFCL_DATA / f"BFCL_v4_{sub}.json"
        if not f.exists():
            groups.append({"name": sub, "total": 0, "samples": [],
                           "note": f"未找到题库文件 {f.name}"})
            continue
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        samples = []
        for line in lines[:n]:
            e = json.loads(line)
            turns = [m["content"] for t in e.get("question", []) for m in t
                     if m.get("role") == "user"]
            samples.append({
                "id": e.get("id", ""),
                "题目": " / ".join(turns)[:400],
                "可用工具": "、".join(fn["name"] for fn in e.get("function", []))[:200],
            })
        groups.append({"name": sub, "total": len(lines), "samples": samples})
    return 200, {"source": "仓库内置题库", "groups": groups}


def _preview_cached(dataset: str, n: int) -> tuple[int, dict]:
    """其他基准只读 modelscope 本地缓存，绝不触发下载。

    缓存是 Arrow 格式，主环境没 pyarrow —— 借 .venv-eval 的 python 跑一下。
    """
    needle = re.sub(r"[^a-z0-9]", "", dataset.lower())
    hits = [d for d in MS_CACHE.glob("*__*")
            if d.is_dir() and needle in re.sub(r"[^a-z0-9]", "", d.name.lower())]
    arrows = sorted({f for d in hits for f in d.rglob("*.arrow")})
    if not arrows:
        with LOCK:
            st = dict(DOWNLOADS.get(dataset, {}))
        if st.get("status") == "running":
            note = "正在下载数据集…下好会自动展示样题。"
        elif st.get("status") == "error":
            note = f"下载失败：{st.get('error', '未知错误')}"
        else:
            note = "本地还没有这个基准的数据集。先点下面的按钮下载，下好就能预览，再发起评测。"
        return 200, {"source": "", "groups": [], "note": note,
                     "download": st.get("status", ""), "downloadable": True}
    py = EVALSCOPE.parent / "python"
    script = (
        "import sys, json, pyarrow as pa, pyarrow.ipc\n"
        "out = []\n"
        "for p in sys.argv[2:]:\n"
        "    with pa.memory_map(p) as src:\n"
        "        try: t = pa.ipc.open_stream(src).read_all()\n"
        "        except pa.lib.ArrowInvalid: t = pa.ipc.open_file(src).read_all()\n"
        "    n = min(int(sys.argv[1]), t.num_rows)\n"
        "    rows = [{c: str(t.column(c)[i].as_py())[:400] for c in t.column_names}\n"
        "            for i in range(n)]\n"
        "    out.append({'file': p.rsplit('/', 1)[-1], 'total': t.num_rows, 'rows': rows})\n"
        "print(json.dumps(out, ensure_ascii=False))\n")
    proc = subprocess.run([str(py), "-c", script, str(n)] + [str(a) for a in arrows[:4]],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return 500, {"error": f"读缓存失败: {proc.stderr.strip()[:300]}"}
    groups = [{"name": g["file"].removesuffix(".arrow"), "total": g["total"],
               "samples": g["rows"]} for g in json.loads(proc.stdout)]
    return 200, {"source": "modelscope 本地缓存", "groups": groups}


def preview_dataset(body: dict) -> tuple[int, dict]:
    dataset = (body.get("dataset") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", dataset or ""):
        return 400, {"error": f"基准名不合法: {dataset!r}"}
    n = min(max(int(body.get("n", 5)), 1), 20)
    if dataset == "bfcl_v4":
        subsets = [s for s in body.get("subsets", []) if re.fullmatch(r"[\w\-]+", s)]
        if not subsets:
            subsets = next(b["subsets"] for b in BENCHMARKS if b["name"] == "bfcl_v4")
        return _preview_bfcl(subsets, n)
    return _preview_cached(dataset, n)


def _download_worker(dataset: str) -> None:
    """借 EvalScope 自己的 adapter.load() 拉数据集——和评测时下载的是同一份缓存。"""
    py = EVALSCOPE.parent / "python"
    script = (
        "import sys\n"
        "from evalscope import TaskConfig\n"
        "from evalscope.api.registry import get_benchmark\n"
        "import evalscope.benchmarks\n"  # 触发基准注册
        "name = sys.argv[1]\n"
        "get_benchmark(name, TaskConfig(model='dummy', datasets=[name])).load()\n")
    try:
        proc = subprocess.run([str(py), "-c", script, dataset],
                              capture_output=True, text=True, timeout=1800)
        err = "" if proc.returncode == 0 else \
            (proc.stderr.strip().splitlines() or ["未知错误"])[-1][:300]
    except subprocess.TimeoutExpired:
        err = "下载超时（30 分钟），可能是网络问题或数据集太大"
    except Exception as e:  # noqa: BLE001
        err = str(e)[:300]
    with LOCK:
        DOWNLOADS[dataset] = {"status": "error" if err else "done", "error": err}


def download_dataset(body: dict) -> tuple[int, dict]:
    """预览发现没缓存时，先把数据集下到本地——先看菜单，再点菜，再上桌。"""
    dataset = (body.get("dataset") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", dataset or ""):
        return 400, {"error": f"基准名不合法: {dataset!r}"}
    with LOCK:
        if DOWNLOADS.get(dataset, {}).get("status") == "running":
            return 200, {"status": "running"}
        DOWNLOADS[dataset] = {"status": "running", "error": ""}
    threading.Thread(target=_download_worker, args=(dataset,), daemon=True).start()
    return 200, {"status": "running"}


def build_matrix(body: dict) -> tuple[int, dict]:
    with LOCK:
        done = [(n, j["workdir"]) for n, j in JOBS.items() if j["status"] == "done"]
    if not done:
        return 400, {"error": "还没有跑完的评测"}
    # 列顺序跟 configs 里的 worker 顺序保持一致（即 router slot 顺序）
    order = {w["name"]: i for i, w in enumerate(WORKERS)}
    done.sort(key=lambda kv: order.get(kv[0], 99))
    out_pred = OUT_DIR / "evalscope_predictions.jsonl"
    out_matrix = OUT_DIR / "evalscope_worker_matrix.csv"
    cmd = [sys.executable, str(ROOT / "eval/evalscope_to_matrix.py"),
           "--out-predictions", str(out_pred), "--out-matrix", str(out_matrix),
           "--on-failure", body.get("on_failure", "exclude")]
    for name, workdir in done:
        cmd += ["--run", f"{name}={workdir}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        return 500, {"error": proc.stderr.strip() or proc.stdout.strip()}
    rows = [line.split(",") for line in
            out_matrix.read_text(encoding="utf-8").splitlines() if line]
    return 200, {"log": proc.stdout.strip(), "dropped": proc.stderr.strip(),
                 "header": rows[0], "rows": rows[1:501], "total_rows": len(rows) - 1,
                 "predictions": str(out_pred), "matrix": str(out_matrix),
                 "train_cmd": ("python train/train_trinity_bfcl.py --config configs/bfcl.yaml"
                               f" --predictions {out_pred.relative_to(ROOT)}")}


def status_payload() -> dict:
    with LOCK:
        jobs = {n: dict(j) for n, j in JOBS.items()}
    for job in jobs.values():
        pct, tail = tail_progress(Path(job["log"]))
        job["progress"] = 100 if job["status"] == "done" else pct
        job["tail"] = tail
    return {"jobs": [jobs[n] for n in jobs],
            "running": any(j["status"] == "running" for j in jobs.values())}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._send(200, {
                "workers": [{**w, "key_set": bool(not w["key_env"] or
                                                  os.environ.get(w["key_env"]))}
                            for w in WORKERS],
                "benchmarks": BENCHMARKS,
                "evalscope_ready": EVALSCOPE.exists(),
            })
        elif path == "/api/status":
            self._send(200, status_payload())
        elif path == "/api/log":
            name = (re.search(r"worker=([^&]+)", self.path) or [None, ""])[1]
            job = JOBS.get(name.replace("%20", " ") if name else "")
            if not job:
                self._send(404, {"error": "no such worker"}); return
            self._send(200, Path(job["log"]).read_bytes(), "text/plain; charset=utf-8")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/start":
                code, resp = start_runs(body)
            elif self.path == "/api/matrix":
                code, resp = build_matrix(body)
            elif self.path == "/api/retry":
                code, resp = retry_failed(body)
            elif self.path == "/api/workers":
                code, resp = edit_workers(body)
            elif self.path == "/api/key":
                code, resp = inject_key(body)
            elif self.path == "/api/test":
                code, resp = test_worker(body)
            elif self.path == "/api/preview":
                code, resp = preview_dataset(body)
            elif self.path == "/api/download":
                code, resp = download_dataset(body)
            else:
                code, resp = 404, {"error": "not found"}
            self._send(code, resp)
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):       # quiet
        pass


def main():
    ap = argparse.ArgumentParser(description="Web console for the unified evaluator.")
    ap.add_argument("--config", default="configs/bfcl.yaml",
                    help="worker 清单来源（缺省回退 configs/bfcl.example.yaml）")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        cfg = ROOT / "configs/bfcl.example.yaml"
        print(f"[eval-ui] {args.config} 不存在，回退 {cfg}", flush=True)
    load_config(cfg)
    print(f"[eval-ui] workers: {[w['name'] for w in WORKERS]}", flush=True)
    print(f"[eval-ui] evalscope: {EVALSCOPE} ({'OK' if EVALSCOPE.exists() else '缺失，先跑 scripts/setup_eval_env.sh'})", flush=True)
    # 只绑本机：这个服务能发起子进程、能读 API key，不应暴露到局域网
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[eval-ui] http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
