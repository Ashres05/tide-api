"""
Lightweight in-process job manager for long-running admin endpoints.

Phase 1 goal: get the long-running refresh/backfill work off the HTTP request
thread so the client (cron, browser, reverse proxy) no longer times out. Jobs
run in a small background thread pool; status is polled via GET /v1/jobs/{id}.

Deliberately not using Celery/RQ yet — the API runs as a single uvicorn
process on one EC2 box, and the weekly cron dispatches at most one job at a
time. If/when we scale out, swap this module for a Redis-backed queue; the
endpoint surface does not need to change.
"""
from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


JobStatus = str  # "pending" | "running" | "succeeded" | "failed"

API_KEY_HEADER = "X-API-Key"
_API_KEY_ENV = "TIDE_API_KEY"
_API_KEY_WARNED = False


@dataclass
class Job:
    """Snapshot of a background job's state."""
    id: str
    name: str
    status: JobStatus = "pending"
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Any = None
    last_log: Optional[str] = None
    # Coarse-grained progress label, updated by the running target via set_step().
    step: Optional[str] = None
    # Rolling log of (timestamp, step, elapsed_since_prev_sec). Useful for
    # diagnosing which stage is slow — polled via GET /v1/jobs/{id}.
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop internal monotonic-anchor keys (prefixed with "_") from steps
        # so GET /v1/jobs/{id} returns a clean payload.
        d["steps"] = [
            {k: v for k, v in s.items() if not k.startswith("_")}
            for s in d.get("steps", [])
        ]
        return d


class JobAlreadyRunningError(RuntimeError):
    """Raised when a job with the same name is already active."""


class JobManager:
    """
    Thread-safe registry of background jobs.

    - Jobs are identified by a name (e.g. "refresh_weekly"). A name-level
      lock prevents duplicate concurrent runs, which matters because the
      training pipeline writes shared artifact files.
    - Terminal jobs are retained in memory for polling, up to `retain`.
    """

    def __init__(self, max_workers: int = 2, retain: int = 50) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tide-job"
        )
        self._jobs: Dict[str, Job] = {}
        self._futures: Dict[str, Future] = {}
        self._active_names: set[str] = set()
        self._lock = threading.Lock()
        self._retain = retain

    def submit(self, name: str, target: Callable[..., Any], *args: Any, **kwargs: Any) -> Job:
        """Queue `target` to run in the background. Returns the initial Job snapshot."""
        with self._lock:
            if name in self._active_names:
                raise JobAlreadyRunningError(f"Job '{name}' is already running.")
            job = Job(
                id=str(uuid.uuid4()),
                name=name,
                status="pending",
                created_at=_utcnow(),
            )
            self._jobs[job.id] = job
            self._active_names.add(name)
            self._gc_locked()
            future = self._executor.submit(self._run, job.id, name, target, args, kwargs)
            self._futures[job.id] = future
        return _snapshot(job)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return _snapshot(job) if job is not None else None

    def list(self, limit: int = 20) -> List[Job]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )[:limit]
            return [_snapshot(j) for j in jobs]

    def is_running(self, name: str) -> bool:
        with self._lock:
            return name in self._active_names

    def set_step(self, job_id: str, step: str) -> None:
        """
        Append a step marker to the job. Records wall time and elapsed since
        the previous step so a poller can read which stage is slow without
        needing access to the server log file.
        """
        now = time.perf_counter()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            prev_elapsed: Optional[float] = None
            if job.steps:
                last_ts = job.steps[-1].get("_perf", now)
                prev_elapsed = round(now - last_ts, 2)
            entry: Dict[str, Any] = {
                "step": step,
                "at": _utcnow(),
                "elapsed_prev_sec": prev_elapsed,
                # Monotonic anchor for the NEXT step's elapsed calc. Stored
                # under a leading underscore so clients can ignore it.
                "_perf": now,
            }
            job.steps.append(entry)
            job.step = step
            job.last_log = f"Step: {step}"
        logger.info("job %s step: %s", job_id, step)

    def _run(
        self,
        job_id: str,
        name: str,
        target: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        # Publish the job id to the thread so nested code can call set_step().
        _current_job_id.set(job_id)
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.started_at = _utcnow()
            job.last_log = f"Starting job '{name}'..."
        logger.info("job %s: starting %s", job_id, name)
        try:
            result = target(*args, **kwargs)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "succeeded"
                job.finished_at = _utcnow()
                job.result = result
                job.last_log = f"Job '{name}' succeeded."
            logger.info("job %s: succeeded %s", job_id, name)
        except Exception as e:
            # Intentionally broad: we must never leak the active-name lock or
            # let an uncaught exception propagate into the worker thread.
            logger.exception("job %s: failed %s", job_id, name)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.finished_at = _utcnow()
                job.error = f"{type(e).__name__}: {e}"
                job.last_log = f"Job '{name}' failed: {e}"
        finally:
            with self._lock:
                self._active_names.discard(name)

    def _gc_locked(self) -> None:
        """Drop oldest terminal jobs when the retention cap is exceeded."""
        if len(self._jobs) <= self._retain:
            return
        terminal_states = {"succeeded", "failed"}
        terminal = [j for j in self._jobs.values() if j.status in terminal_states]
        terminal.sort(key=lambda j: j.created_at)
        drop = len(self._jobs) - self._retain
        for j in terminal[:drop]:
            self._jobs.pop(j.id, None)
            self._futures.pop(j.id, None)


def _snapshot(job: Job) -> Job:
    """Return an independent copy of a Job so callers can't mutate internal state."""
    return Job(**asdict(job))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()

# Thread-local handle to the currently-running job. Set by JobManager._run and
# read by set_step() so background tasks can emit progress markers without
# having to receive the job id as a parameter.
_current_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_job_id", default=None
)


def get_manager() -> JobManager:
    """Module-level singleton. Safe to call from any request thread."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = JobManager()
    return _manager


def set_step(step: str) -> None:
    """
    Record a progress step on the currently-running job.

    No-op when called outside a job (e.g. during a direct script run or a
    unit test), so background-task code can safely emit steps regardless of
    how it's invoked.
    """
    job_id = _current_job_id.get()
    if job_id is None:
        return
    get_manager().set_step(job_id, step)


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """
    FastAPI dependency that guards admin endpoints with a shared secret.

    If `TIDE_API_KEY` is unset the dependency is permissive — this keeps local
    dev ergonomic — but logs a single warning so you don't accidentally run
    prod without it. Set `TIDE_API_KEY` in the EC2 environment before exposing
    the API publicly.
    """
    expected = os.environ.get(_API_KEY_ENV, "").strip()
    if not expected:
        global _API_KEY_WARNED
        if not _API_KEY_WARNED:
            logger.warning(
                "%s is not set; admin endpoints are open. "
                "Set %s in the environment before exposing the API.",
                _API_KEY_ENV,
                _API_KEY_ENV,
            )
            _API_KEY_WARNED = True
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
