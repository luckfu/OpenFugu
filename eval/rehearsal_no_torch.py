#!/usr/bin/env python3
"""Torch-free project rehearsal.

Boots the real OpenAI-compatible server with --mock-router, sends one chat
completion request, and verifies that the Coordinator loop returned a response.
This proves the serving surface and orchestration wiring without training,
downloading torch, or loading any model weights.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


def wait_ready(port, proc, timeout=30):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"[rehearsal] server exited early ({proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise SystemExit("[rehearsal] server did not become ready")


def post_chat(port, question):
    body = json.dumps({"messages": [{"role": "user", "content": question}]}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description="Torch-free OpenFugu rehearsal.")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--question", default="flatten a nested list in one line")
    ap.add_argument("--serve-script", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "openfugu", "serve.py"))
    args = ap.parse_args()

    cmd = [sys.executable, args.serve_script, "--mock-router", "--port", str(args.port)]
    print(f"[rehearsal] booting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        wait_ready(args.port, proc)
        resp = post_chat(args.port, args.question)
        content = resp["choices"][0]["message"]["content"]
        turns = resp.get("usage", {}).get("fugu_turns", 0)
        print(f"[rehearsal] turns={turns}", flush=True)
        print(f"[rehearsal] content={content!r}", flush=True)
        if turns > 0 and content:
            print("PASS — torch-free serve rehearsal completed")
            return 0
        print("FAIL — empty response or no coordinator turns")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
