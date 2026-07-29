# llmops/system_stats.py
"""System resource stats for the monitoring dashboard (System Health tab).

Mirrors the reference ML-Data-Ops dashboard's system panel. Uses psutil
when available; degrades to 'n/a' rather than crashing if it is not.
"""
from __future__ import annotations

import os
import platform
import shutil
from typing import Any


def system_stats() -> dict[str, Any]:
    """CPU / memory / disk usage + host info. Never raises."""
    stats: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "cpu_percent": None,
        "memory_percent": None,
        "disk_percent": None,
        "psutil": False,
    }
    try:
        import psutil  # optional

        stats.update(
            {
                "cpu_percent": psutil.cpu_percent(interval=0.3),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(os.sep).percent,
                "cpu_count": psutil.cpu_count(logical=True),
                "psutil": True,
            }
        )
    except Exception:  # noqa: BLE001 - psutil missing or blocked
        # Fallback: disk via stdlib shutil; CPU/mem left as None -> 'n/a'
        try:
            total, used, _ = shutil.disk_usage(os.sep)
            stats["disk_percent"] = round(used / total * 100, 1)
        except Exception:  # noqa: BLE001
            pass
    return stats


def db_file_sizes(paths: dict[str, str]) -> list[dict[str, Any]]:
    """Sizes (KB) of the metrics/cache SQLite files, for the ops panel."""
    out = []
    for label, path in paths.items():
        size_kb = (
            round(os.path.getsize(path) / 1024, 1) if os.path.isfile(path) else 0.0
        )
        out.append({"store": label, "path": path, "size_kb": size_kb})
    return out


def kb_stats(tickets_csv: str, index_dir: str, sops_dir: str) -> dict[str, Any]:
    """Knowledge-base summary for the sidebar panel: ticket/SOP counts and
    build times. Never raises (missing files -> zeros / None)."""
    import csv
    import glob
    import time as _time

    def _mtime(path: str) -> str | None:
        if not os.path.exists(path):
            return None
        return _time.strftime("%Y-%m-%d %H:%M",
                              _time.localtime(os.path.getmtime(path)))

    tickets = 0
    if os.path.isfile(tickets_csv):
        try:
            with open(tickets_csv, "r", encoding="utf-8") as fh:
                tickets = max(0, sum(1 for _ in csv.reader(fh)) - 1)
        except Exception:  # noqa: BLE001
            tickets = 0
    index_path = os.path.join(index_dir, "index.faiss")
    sop_count = len(glob.glob(os.path.join(sops_dir, "*.md")))
    return {
        "tickets_indexed": tickets,
        "index_built": _mtime(index_path),
        "sop_pages": sop_count,
        "sop_built": _mtime(sops_dir),
        "index_exists": os.path.isfile(index_path),
    }
