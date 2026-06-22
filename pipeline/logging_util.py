"""Shared, file-based logging for every pipeline component.

Persistent logs live under the repo's ``logs/`` directory (override with the
``FT_LOGS_DIR`` env var) and are also exposed over HTTP via the ``/api/logs``
endpoints, so they're accessible without shell access:

    logs/server/ui.log              — web UI server (uvicorn + app output)
    logs/server/inference.log       — inference server (:7200)
    logs/training/<niche>_<run>.log — one file per training run (worker output)
    logs/export/<niche>.log         — one file per export/merge

Pure stdlib (os / sys), so this is platform-agnostic and pulls in no Mac- or
Linux-specific dependency.
"""

import os
import sys

# Repo root = parent of this file's directory (pipeline/).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.environ.get("FT_LOGS_DIR", os.path.join(_ROOT, "logs"))


def _ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def training_log_path(niche: str, run_id: str) -> str:
    return os.path.join(_ensure(os.path.join(LOGS_DIR, "training")), f"{niche}_{run_id}.log")


def export_log_path(niche: str) -> str:
    return os.path.join(_ensure(os.path.join(LOGS_DIR, "export")), f"{niche}.log")


def server_log_path(name: str) -> str:
    return os.path.join(_ensure(os.path.join(LOGS_DIR, "server")), f"{name}.log")


class _Tee:
    """Mirror a text stream to a file as well as the original stream."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_server_logging(name: str) -> str:
    """Tee this process's stdout+stderr into logs/server/<name>.log.

    Capturing at the stream level (rather than configuring uvicorn's logging)
    reliably records both uvicorn request logs and the modules' plain print()
    output, regardless of how the server was launched. Returns the log path.
    """
    path = server_log_path(name)
    fh = open(path, "a", buffering=1)  # line-buffered
    fh.write(f"\n{'='*60}\n=== {name} server started (pid {os.getpid()}) ===\n")
    fh.flush()
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return path


def list_logs() -> list:
    """List all log files under LOGS_DIR as {path, size, modified} (newest first)."""
    out = []
    for root, _dirs, files in os.walk(LOGS_DIR):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            out.append({
                "path": os.path.relpath(full, LOGS_DIR),
                "size": st.st_size,
                "modified": st.st_mtime,
            })
    out.sort(key=lambda e: e["modified"], reverse=True)
    return out


def read_log_tail(rel_path: str, lines: int = 400) -> str:
    """Return the last `lines` lines of a log file. The path is constrained to
    LOGS_DIR (no traversal outside it) — raises ValueError / FileNotFoundError."""
    full = os.path.normpath(os.path.join(LOGS_DIR, rel_path))
    if os.path.commonpath([os.path.abspath(full), os.path.abspath(LOGS_DIR)]) != os.path.abspath(LOGS_DIR):
        raise ValueError("path escapes logs directory")
    if not os.path.isfile(full):
        raise FileNotFoundError(rel_path)
    with open(full, errors="replace") as f:
        return "".join(f.readlines()[-lines:])
