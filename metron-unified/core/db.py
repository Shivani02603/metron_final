"""
Core DB layer — SQLite persistence for run history and job recovery.

Fix 28: Stores completed run results so regression comparisons are possible.
Fix 21: On server startup, recently completed jobs are re-populated into the
        in-memory job store so results survive a server restart.

Schema:
  runs(run_id TEXT PK, project_id TEXT, timestamp TEXT, health_score REAL,
       domain TEXT, application_type TEXT, status TEXT, results_json TEXT)

DB path defaults to ./metron_runs.db; override via METRON_DB_PATH env var.
"""

from __future__ import annotations
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

_DB_PATH_DEFAULT = "./metron_runs.db"
_lock = threading.Lock()


def _db_path() -> str:
    return os.environ.get("METRON_DB_PATH", _DB_PATH_DEFAULT)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at server startup."""
    with _lock:
        conn = _connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id           TEXT PRIMARY KEY,
                    project_id       TEXT NOT NULL,
                    timestamp        TEXT NOT NULL,
                    health_score     REAL,
                    domain           TEXT,
                    application_type TEXT,
                    status           TEXT DEFAULT 'completed',
                    results_json     TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp)")
            conn.commit()
        finally:
            conn.close()


def save_run(
    run_id: str,
    project_id: str,
    health_score: float,
    domain: str,
    application_type: str,
    results: Dict[str, Any],
    status: str = "completed",
) -> None:
    """Persist a completed run to SQLite."""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, project_id, timestamp, health_score, domain, application_type, status, results_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    project_id,
                    datetime.utcnow().isoformat(),
                    health_score,
                    domain,
                    application_type,
                    status,
                    json.dumps(results),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single run by run_id. Returns None if not found."""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            if d.get("results_json"):
                try:
                    d["results"] = json.loads(d.pop("results_json"))
                except Exception:
                    d.pop("results_json", None)
            return d
        finally:
            conn.close()


def get_runs_for_project(project_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return all runs for a project, sorted newest-first.
    results_json is NOT decoded (summary only) — call get_run() for full results.
    """
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT run_id, project_id, timestamp, health_score, domain,
                       application_type, status
                FROM runs
                WHERE project_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (project_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def compare_runs(run_id_a: str, run_id_b: str) -> Dict[str, Any]:
    """
    Basic diff between two runs: health score delta and per-class pass-rate change.
    Returns a summary dict suitable for the /api/runs/{a}/compare/{b} endpoint.
    """
    a = get_run(run_id_a)
    b = get_run(run_id_b)

    if not a or not b:
        missing = []
        if not a:
            missing.append(run_id_a)
        if not b:
            missing.append(run_id_b)
        return {"error": f"Run(s) not found: {missing}"}

    res_a = a.get("results", {})
    res_b = b.get("results", {})

    health_a = a.get("health_score", 0.0)
    health_b = b.get("health_score", 0.0)

    class_diff: Dict[str, Any] = {}
    classes_a = res_a.get("test_classes", {})
    classes_b = res_b.get("test_classes", {})
    all_classes = set(classes_a) | set(classes_b)

    for cls in sorted(all_classes):
        pr_a = classes_a.get(cls, {}).get("pass_rate")
        pr_b = classes_b.get(cls, {}).get("pass_rate")
        class_diff[cls] = {
            "run_a_pass_rate": pr_a,
            "run_b_pass_rate": pr_b,
            "delta": round((pr_b or 0.0) - (pr_a or 0.0), 4) if pr_a is not None and pr_b is not None else None,
        }

    return {
        "run_id_a":     run_id_a,
        "run_id_b":     run_id_b,
        "timestamp_a":  a.get("timestamp"),
        "timestamp_b":  b.get("timestamp"),
        "health_a":     health_a,
        "health_b":     health_b,
        "health_delta": round(health_b - health_a, 4),
        "class_diff":   class_diff,
    }


def load_recent_jobs(hours: int = 24) -> List[Dict[str, Any]]:
    """
    Fetch runs completed within the last N hours for in-memory job store re-population.
    Used on server startup (Fix 21) so GET /api/job/{id}/results still works after restart.
    """
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT run_id, status, results_json
                FROM runs
                WHERE timestamp >= ? AND status = 'completed'
                ORDER BY timestamp DESC
                LIMIT 200
                """,
                (cutoff,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d.get("results_json"):
                    try:
                        d["results"] = json.loads(d.pop("results_json"))
                    except Exception:
                        d.pop("results_json", None)
                result.append(d)
            return result
        finally:
            conn.close()
