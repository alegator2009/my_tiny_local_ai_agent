from __future__ import annotations

import subprocess
import time
from typing import Any

MAX_OUTPUT_CHARS = 12000


def run_terminal_command(command: str, cwd: str | None = None, timeout_sec: int = 25) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        stdout = (proc.stdout or "")[:MAX_OUTPUT_CHARS]
        stderr = (proc.stderr or "")[:MAX_OUTPUT_CHARS]
        return {
            "ok": proc.returncode == 0,
            "command": command,
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_ms": elapsed_ms,
            "truncated": len(proc.stdout or "") > MAX_OUTPUT_CHARS or len(proc.stderr or "") > MAX_OUTPUT_CHARS,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        stdout = (exc.stdout or "")[:MAX_OUTPUT_CHARS]
        stderr = (exc.stderr or "")[:MAX_OUTPUT_CHARS]
        return {
            "ok": False,
            "command": command,
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_ms": elapsed_ms,
            "error": f"timeout after {timeout_sec}s",
            "truncated": True,
        }
