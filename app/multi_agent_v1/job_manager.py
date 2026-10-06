"""Bounded, FIFO background job runner for POST /multi_agent_v1/jobs (see
app/main.py) -- the asynchronous job-mode wrapper around the EXACT
existing multi_agent_v1 pipeline (via `run_multi_agent_v1_analysis()` in
job_runner.py, never duplicated or changed here).

Render Free/Starter runs this app as a single process on a small, fixed
CPU/RAM budget -- a second full plan analysis running at the same time as
a first would compete for the same limited resources and could push the
instance over its memory limit. `JobManager` enforces "at most
GASWATERAI_JOB_MAX_CONCURRENCY plan analyses in flight at once" (default
1) with a small fixed pool of daemon worker threads pulling from one FIFO
`queue.Queue` -- no new process, no external dependency, just a hard cap
on how much of this one process's own pipeline work can run concurrently.

A submitted job's worker thread runs strictly AFTER the original
POST /multi_agent_v1/jobs request has already returned its 202 response
and ended -- that request's own `timing_diagnostics` TimingRecorder
(a contextvar-scoped object, see timing_diagnostics.py) no longer exists
by the time a worker thread picks the job up. So each worker opens and
closes its OWN timing_diagnostics request scope around the pipeline call
-- mirroring exactly what the synchronous /multi_agent_v1/analyze route
already does -- rather than trying to inherit a context that has already
ended. On top of that per-request diagnostics line, this module logs one
additional structured line per job (`queue_wait_ms`, `processing_
duration_ms`, `total_job_duration_ms`), gated by the same
GASWATERAI_TIMING_DIAGNOSTICS=1 env var and the same
"gaswaterai.timing_diagnostics" logger -- never a new logger, never
customer/plan content.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import timing_diagnostics
from .job_runner import InvalidPdfError, PdfAnalysisError, run_multi_agent_v1_analysis
from .job_store import COMPLETED, FAILED, PROCESSING, QUEUED, InMemoryJobStore, JobRecord, JobStore
from .provider import AgentModelProvider

JOB_MAX_CONCURRENCY_ENV_VAR = "GASWATERAI_JOB_MAX_CONCURRENCY"
DEFAULT_JOB_MAX_CONCURRENCY = 1

JOB_MAX_QUEUE_LENGTH_ENV_VAR = "GASWATERAI_JOB_MAX_QUEUE_LENGTH"
DEFAULT_JOB_MAX_QUEUE_LENGTH = 20

JOB_TTL_SECONDS_ENV_VAR = "GASWATERAI_JOB_TTL_SECONDS"
DEFAULT_JOB_TTL_SECONDS = 3600  # 1 hour -- old completed/failed records are cleaned up after this

_SHUTDOWN = object()


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def max_job_concurrency() -> int:
    return _env_int(JOB_MAX_CONCURRENCY_ENV_VAR, DEFAULT_JOB_MAX_CONCURRENCY)


def max_queue_length() -> int:
    return _env_int(JOB_MAX_QUEUE_LENGTH_ENV_VAR, DEFAULT_JOB_MAX_QUEUE_LENGTH)


def job_ttl_seconds() -> int:
    return _env_int(JOB_TTL_SECONDS_ENV_VAR, DEFAULT_JOB_TTL_SECONDS)


class QueueFullError(RuntimeError):
    """Raised by `JobManager.submit()` when the queue is already at
    GASWATERAI_JOB_MAX_QUEUE_LENGTH -- maps to HTTP 503 in app/main.py."""


class _WorkItem:
    __slots__ = ("job_id", "pdf_bytes", "provider", "filename", "queued_at_perf")

    def __init__(self, job_id: str, pdf_bytes: bytes, provider: AgentModelProvider, filename: Optional[str]):
        self.job_id = job_id
        self.pdf_bytes = pdf_bytes
        self.provider = provider
        self.filename = filename
        self.queued_at_perf = time.perf_counter()


def _log_job_timing(job_id: str, queue_wait_ms: float, processing_ms: float, final_status: str) -> None:
    if not timing_diagnostics.enabled():
        return
    payload = {
        "job_id": job_id,
        "queue_wait_ms": round(queue_wait_ms, 2),
        "processing_duration_ms": round(processing_ms, 2),
        "total_job_duration_ms": round(queue_wait_ms + processing_ms, 2),
        "status": final_status,
    }
    timing_diagnostics.logger.info("multi_agent_v1_job_timing %s", json.dumps(payload))


class JobManager:
    """Owns one FIFO queue and a fixed pool of daemon worker threads. Not
    a singleton by itself -- see `get_job_manager()`/
    `reset_job_manager_for_tests()` below for the process-wide instance
    app/main.py actually uses."""

    def __init__(self, store: Optional[JobStore] = None, max_concurrency: Optional[int] = None):
        self.store: JobStore = store if store is not None else InMemoryJobStore()
        self._max_concurrency = max_concurrency if max_concurrency is not None else max_job_concurrency()
        self._queue: "queue.Queue" = queue.Queue()
        self._submit_lock = threading.Lock()  # guards the idempotency-check-then-enqueue sequence
        self._workers: list[threading.Thread] = []
        for i in range(self._max_concurrency):
            t = threading.Thread(target=self._worker_loop, name=f"gaswaterai-job-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def submit(
        self, pdf_bytes: bytes, provider: AgentModelProvider, fingerprint: str, filename: Optional[str] = None,
    ) -> tuple[str, bool]:
        """Returns (job_id, created). created=False when an identical
        input (same document_fingerprint) already has a QUEUED/PROCESSING
        job -- the caller gets that job's id back instead of a second,
        costly analysis being started (idempotency, job-mode spec
        section 7)."""
        self._cleanup_expired()
        with self._submit_lock:
            existing = self.store.find_active_by_fingerprint(fingerprint)
            if existing is not None:
                return existing.job_id, False
            if self._queue.qsize() >= max_queue_length():
                raise QueueFullError(f"Job queue is full (max {max_queue_length()}). Try again later.")

            job_id = uuid.uuid4().hex
            record = JobRecord(
                job_id=job_id, fingerprint=fingerprint, status=QUEUED, created_at=datetime.now(timezone.utc),
            )
            self.store.create(record)
            self._queue.put(_WorkItem(job_id, pdf_bytes, provider, filename))
            return job_id, True

    def get(self, job_id: str):
        self._cleanup_expired()
        return self.store.get(job_id)

    def shutdown(self) -> None:
        """Test-only: stops every worker thread cleanly (poison pill) so a
        discarded JobManager does not leak daemon threads blocked forever
        on an empty queue across many test runs."""
        for _ in self._workers:
            self._queue.put(_SHUTDOWN)
        for t in self._workers:
            t.join(timeout=5)

    def _cleanup_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=job_ttl_seconds())
        self.store.delete_completed_before(cutoff)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._queue.task_done()
                break
            try:
                self._process(item)
            finally:
                self._queue.task_done()

    def _process(self, item: _WorkItem) -> None:
        queue_wait_ms = (time.perf_counter() - item.queued_at_perf) * 1000
        started_at = datetime.now(timezone.utc)
        self.store.update(item.job_id, status=PROCESSING, started_at=started_at)

        timing_token = timing_diagnostics.start_request()
        t0 = time.perf_counter()
        try:
            result = run_multi_agent_v1_analysis(item.pdf_bytes, item.provider, filename=item.filename)
        except (InvalidPdfError, PdfAnalysisError) as exc:
            self._finish_failed(item.job_id, str(exc), queue_wait_ms, t0)
            return
        except Exception as exc:  # noqa: BLE001 -- a worker thread must never die silently; surface a safe, technical message only
            self._finish_failed(item.job_id, f"{type(exc).__name__}: analysis failed", queue_wait_ms, t0)
            return
        finally:
            timing_diagnostics.finish_and_log()
            timing_diagnostics.end_request(timing_token)

        processing_ms = (time.perf_counter() - t0) * 1000
        self.store.update(item.job_id, status=COMPLETED, result=result, completed_at=datetime.now(timezone.utc))
        _log_job_timing(item.job_id, queue_wait_ms, processing_ms, COMPLETED)

    def _finish_failed(self, job_id: str, error: str, queue_wait_ms: float, t0: float) -> None:
        processing_ms = (time.perf_counter() - t0) * 1000
        self.store.update(job_id, status=FAILED, error=error, completed_at=datetime.now(timezone.utc))
        _log_job_timing(job_id, queue_wait_ms, processing_ms, FAILED)


_singleton_lock = threading.Lock()
_singleton: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = JobManager()
        return _singleton


def reset_job_manager_for_tests() -> None:
    """Test-only: shuts down and drops the process-wide JobManager so the
    next `get_job_manager()` call builds a fresh one -- needed because a
    running JobManager's worker-thread count is fixed at construction
    time, but tests need to vary GASWATERAI_JOB_MAX_CONCURRENCY (and the
    other env-driven limits) between test cases."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.shutdown()
        _singleton = None
