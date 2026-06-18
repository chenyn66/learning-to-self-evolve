import os
import sqlite3
from typing import List, Tuple, Optional
from lse.paths import bird_db_root, bird_ground_truth_cache_path

# from func_timeout import func_timeout, FunctionTimedOut
import time


# Default search roots for SQLite databases (Mini-Dev and Train)
_DEFAULT_DB_ROOT_DEV = str(bird_db_root("dev"))
_DEFAULT_DB_ROOT_TRAIN = str(bird_db_root("train"))

SQL_TIMEOUT = float(os.environ.get("SQL_TIMEOUT", 30))


# setup sql cache

from collections import OrderedDict
import threading
import logging

# In-memory, process-wide cache controls
SQL_CACHE_ENABLED = os.environ.get("SQL_CACHE", "1") not in ("0", "false", "False")
logging.info(f"SQL_CACHE_ENABLED: {SQL_CACHE_ENABLED}")
SQL_CACHE_MAXSIZE = int(os.environ.get("SQL_CACHE_MAXSIZE", "100000"))

_SQL_RESULT_CACHE = OrderedDict()
_SQL_CACHE_LOCK = threading.RLock()

if SQL_CACHE_ENABLED:
    import pickle
    cache_path = bird_ground_truth_cache_path()
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            _SQL_RESULT_CACHE = pickle.load(f)

def _normalize_sql(sql: str) -> str:
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return " ".join(s.split())

def _cache_key(db_path: str, sql: str):
    return (db_path, _normalize_sql(sql))

def _cache_get(db_path: str, sql: str):
    if not SQL_CACHE_ENABLED:
        return None
    key = _cache_key(db_path, sql)
    with _SQL_CACHE_LOCK:
        val = _SQL_RESULT_CACHE.get(key)
        if val is not None:
            _SQL_RESULT_CACHE.move_to_end(key)
        return val

def _cache_set(db_path: str, sql: str, rows):
    if not SQL_CACHE_ENABLED:
        return
    key = _cache_key(db_path, sql)
    with _SQL_CACHE_LOCK:
        _SQL_RESULT_CACHE[key] = rows
        _SQL_RESULT_CACHE.move_to_end(key)
        if len(_SQL_RESULT_CACHE) > SQL_CACHE_MAXSIZE:
            _SQL_RESULT_CACHE.popitem(last=False)

def clear_sql_cache():
    with _SQL_CACHE_LOCK:
        _SQL_RESULT_CACHE.clear()


def _candidate_roots(split: Optional[str] = None) -> List[str]:
    # Prefer roots by requested split, then fall back to both
    if split == "dev":
        roots: List[str] = [_DEFAULT_DB_ROOT_DEV, _DEFAULT_DB_ROOT_TRAIN]
    elif split == "train":
        roots = [_DEFAULT_DB_ROOT_TRAIN, _DEFAULT_DB_ROOT_DEV]
    else:
        roots = [_DEFAULT_DB_ROOT_DEV, _DEFAULT_DB_ROOT_TRAIN]
    # Also consider repo-relative defaults derived from LSE_BIRD_DATA_ROOT.
    rel_candidates = [_DEFAULT_DB_ROOT_DEV, _DEFAULT_DB_ROOT_TRAIN]
    for p in rel_candidates:
        if p not in roots:
            roots.append(p)
    # Optional: allow env override (colon-separated list)
    env_roots = os.environ.get("BIRD_DB_ROOTS")
    if env_roots:
        for p in env_roots.split(":"):
            p = p.strip()
            if p and p not in roots:
                roots.append(p)
    return roots


def _find_db_path(db_id: str, split: Optional[str] = None) -> str:
    for root in _candidate_roots(split):
        candidate = os.path.join(root, db_id, f"{db_id}.sqlite")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"SQLite database for db_id '{db_id}' not found in roots: {_candidate_roots(split)}"
    )


# def _execute_sql(db_path: str, sql: str) -> List[Tuple]:
#     conn = sqlite3.connect(db_path)
#     try:
#         cursor = conn.cursor()
#         cursor.execute(sql)
#         rows = cursor.fetchall()
#         return rows
#     finally:
#         conn.close()

def _execute_sql(db_path: str, sql: str, timeout: float = SQL_TIMEOUT):
    cached = _cache_get(db_path, sql)
    if cached is not None:
        return cached

    conn = sqlite3.connect(db_path)
    try:
        deadline = time.monotonic() + timeout
        def _progress():
            return 1 if time.monotonic() >= deadline else 0
        conn.set_progress_handler(_progress, 1000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        _cache_set(db_path, sql, rows)
        return rows
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise TimeoutError("query timed out")
        raise
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()

def _calculate_ex(predicted_res: List[Tuple], ground_truth_res: List[Tuple]) -> int:
    return 1 if set(predicted_res) == set(ground_truth_res) else 0


def eval_bird_ex(pred: str, problem: dict, split: Optional[str] = None) -> Tuple[bool, str]:
    """
    Evaluate Execution Accuracy (EX) for a single example, consistent with evaluation_ex.py.

    Args:
        pred: predicted SQL string
        problem: one data entry (e.g., a dict from train.json or mini_dev_sqlite.json)
        split: which split the problem belongs to: "dev", "train", or None for auto

    Returns:
        (ok, err_msg): ok=True if EX=1 (prediction result set equals ground-truth
        result set), else False. err_msg is '' on success; otherwise the error cause.

    Notes:
        - SQLite only. The DB is auto-located under known Mini-Dev/Train roots.
        - Uses a timeout similar to the official evaluator; any error/timeout -> False with message.
    """
    db_id = problem["db_id"]
    gold_sql = problem["SQL"]

    db_path = _find_db_path(db_id, split)

    # Execute predicted SQL with timeout and error capture
    try:
        pred_res = _execute_sql(db_path, pred)
    except TimeoutError:
        print(f"predicted SQL timeout: {pred}")
        return False, "predicted SQL timeout"
    except Exception as e:
        print(f"predicted SQL error: {e}")
        return False, f"predicted SQL error: {e}"

    # Execute ground-truth SQL (should succeed; if not, report error)
    try:
        gold_res = _execute_sql(db_path, gold_sql)
    except TimeoutError:
        print(f"ground-truth SQL timeout: {gold_sql}")
        return False, "ground-truth SQL timeout"
    except Exception as e:
        print(f"ground-truth SQL error: {e}")
        return False, f"ground-truth SQL error: {e}"

    ok = bool(_calculate_ex(pred_res, gold_res))
    if ok:
        return True, ""
    # Both executed but mismatch: include both result sets
    return False, f"result mismatch | pred: {pred_res} | target: {gold_res}"