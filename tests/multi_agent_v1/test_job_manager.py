"""Unit tests for JobManager (job-mode epic): FIFO scheduling, bounded
concurrency, fingerprint-based idempotency, queue-length limit, and TTL
cleanup -- all against a FAKE, monkeypatched pipeline function (never the
real multi_agent_v1 pipeline) so these tests are fast and fully control
timing/failure without a real PDF or model call. Equivalence with the
real pipeline/sync endpoint is covered separately in
tests/test_multi_agent_v1_jobs_api.py."""
from __future__ import annotations

import threading
import time

import pytest

from app.multi_agent_v1 import job_manager as job_manager_module
from app.multi_agent_v1.job_manager import JobManager, QueueFullError
from app.multi_agent_v1.job_store import COMPLETED, FAILED, PROCESSING, QUEUED


@pytest.fixture
def make_manager():
    created: list[JobManager] = []

    def _make(max_concurrency: int = 1) -> JobManager:
        m = JobManager(max_concurrency=max_concurrency)
        created.append(m)
        return m

    yield _make
    for m in created:
        m.shutdown()


def _wait_for_status(manager: JobManager, job_id: str, status: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record is not None and record.status == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach status {status!r} within {timeout}s")


def test_submit_creates_a_queued_job(make_manager, monkeypatch):
    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", lambda *a, **k: {"ok": True})
    manager = make_manager(max_concurrency=1)
    job_id, created = manager.submit(b"irrelevant", provider=None, fingerprint="fp-1")
    assert created is True
    record = manager.get(job_id)
    assert record is not None
    assert record.status in (QUEUED, PROCESSING, COMPLETED)  # a fast fake worker may already be ahead


def test_job_reaches_completed_with_the_fake_result(make_manager, monkeypatch):
    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", lambda *a, **k: {"canonical_plan_understanding": {"x": 1}})
    manager = make_manager(max_concurrency=1)
    job_id, _ = manager.submit(b"irrelevant", provider=None, fingerprint="fp-2")
    _wait_for_status(manager, job_id, COMPLETED)
    record = manager.get(job_id)
    assert record.result == {"canonical_plan_understanding": {"x": 1}}
    assert record.error is None
    assert record.started_at is not None
    assert record.completed_at is not None


def test_job_failure_surfaces_as_failed_with_a_safe_message(make_manager, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("some internal detail with a secret token abc123")

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _boom)
    manager = make_manager(max_concurrency=1)
    job_id, _ = manager.submit(b"irrelevant", provider=None, fingerprint="fp-3")
    _wait_for_status(manager, job_id, FAILED)
    record = manager.get(job_id)
    assert record.result is None
    assert record.error == "RuntimeError: analysis failed"  # never the raw exception text (no secrets/content)


def test_invalid_pdf_error_surfaces_as_failed(make_manager, monkeypatch):
    from app.multi_agent_v1.job_runner import InvalidPdfError

    def _invalid(*a, **k):
        raise InvalidPdfError("Uploaded file is not a PDF.")

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _invalid)
    manager = make_manager(max_concurrency=1)
    job_id, _ = manager.submit(b"irrelevant", provider=None, fingerprint="fp-4")
    _wait_for_status(manager, job_id, FAILED)
    assert manager.get(job_id).error == "Uploaded file is not a PDF."


def test_duplicate_fingerprint_while_active_returns_the_same_job_id(make_manager, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    call_count = {"n": 0}

    def _slow(*a, **k):
        call_count["n"] += 1
        started.set()
        release.wait(timeout=5)
        return {"ok": True}

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _slow)
    manager = make_manager(max_concurrency=1)

    job_id_1, created_1 = manager.submit(b"same-bytes", provider=None, fingerprint="fp-dup")
    assert started.wait(timeout=5)  # make sure it is actually in flight, not just queued

    job_id_2, created_2 = manager.submit(b"same-bytes", provider=None, fingerprint="fp-dup")
    release.set()

    assert created_1 is True
    assert created_2 is False
    assert job_id_1 == job_id_2
    _wait_for_status(manager, job_id_1, COMPLETED)
    assert call_count["n"] == 1  # the identical input was never analyzed twice


def test_different_fingerprints_both_get_their_own_job(make_manager, monkeypatch):
    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", lambda *a, **k: {"ok": True})
    manager = make_manager(max_concurrency=2)
    job_id_1, created_1 = manager.submit(b"a", provider=None, fingerprint="fp-a")
    job_id_2, created_2 = manager.submit(b"b", provider=None, fingerprint="fp-b")
    assert created_1 is True and created_2 is True
    assert job_id_1 != job_id_2


def test_max_concurrency_one_never_runs_two_jobs_at_once(make_manager, monkeypatch):
    lock = threading.Lock()
    active = {"count": 0, "max_seen": 0}

    def _tracked(*a, **k):
        with lock:
            active["count"] += 1
            active["max_seen"] = max(active["max_seen"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return {"ok": True}

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _tracked)
    manager = make_manager(max_concurrency=1)
    job_ids = [manager.submit(b"x", provider=None, fingerprint=f"fp-conc-{i}")[0] for i in range(4)]
    for job_id in job_ids:
        _wait_for_status(manager, job_id, COMPLETED, timeout=10)

    assert active["max_seen"] == 1


def test_max_concurrency_two_allows_two_jobs_at_once(make_manager, monkeypatch):
    lock = threading.Lock()
    active = {"count": 0, "max_seen": 0}
    barrier_hit = threading.Event()

    def _tracked(*a, **k):
        with lock:
            active["count"] += 1
            active["max_seen"] = max(active["max_seen"], active["count"])
            if active["count"] == 2:
                barrier_hit.set()
        time.sleep(0.2)
        with lock:
            active["count"] -= 1
        return {"ok": True}

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _tracked)
    manager = make_manager(max_concurrency=2)
    job_ids = [manager.submit(b"x", provider=None, fingerprint=f"fp-conc2-{i}")[0] for i in range(2)]
    assert barrier_hit.wait(timeout=5)
    for job_id in job_ids:
        _wait_for_status(manager, job_id, COMPLETED, timeout=10)
    assert active["max_seen"] == 2


def test_queue_processes_in_fifo_order(make_manager, monkeypatch):
    lock = threading.Lock()
    order: list[str] = []
    first_release = threading.Event()
    release = threading.Event()

    def _dispatch(pdf_bytes, provider, filename=None):
        # job0 is held back on its own event so job1-3 are guaranteed to
        # still be sitting in the FIFO queue (not yet dispatched) when
        # they are submitted; one single patched function (never swapped
        # mid-test) avoids a race against whichever worker thread picks
        # job0 up first.
        (first_release if filename == "job0" else release).wait(timeout=5)
        with lock:
            order.append(filename)
        return {"ok": True}

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _dispatch)
    manager = make_manager(max_concurrency=1)

    job0, _ = manager.submit(b"x", provider=None, fingerprint="fp-fifo-0", filename="job0")
    time.sleep(0.05)  # let the one worker actually pick job0 up before the rest are queued
    job1, _ = manager.submit(b"x", provider=None, fingerprint="fp-fifo-1", filename="job1")
    job2, _ = manager.submit(b"x", provider=None, fingerprint="fp-fifo-2", filename="job2")
    job3, _ = manager.submit(b"x", provider=None, fingerprint="fp-fifo-3", filename="job3")

    first_release.set()
    release.set()
    for job_id in (job0, job1, job2, job3):
        _wait_for_status(manager, job_id, COMPLETED, timeout=10)

    assert order == ["job0", "job1", "job2", "job3"]


def test_max_queue_length_rejects_overflow(make_manager, monkeypatch):
    block = threading.Event()

    def _blocked(*a, **k):
        block.wait(timeout=5)
        return {"ok": True}

    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", _blocked)
    monkeypatch.setenv(job_manager_module.JOB_MAX_QUEUE_LENGTH_ENV_VAR, "1")
    manager = make_manager(max_concurrency=1)

    job0, _ = manager.submit(b"x", provider=None, fingerprint="fp-q-0")
    time.sleep(0.05)  # job0 is now PROCESSING, queue itself is empty
    manager.submit(b"x", provider=None, fingerprint="fp-q-1")  # fills the one queue slot

    with pytest.raises(QueueFullError):
        manager.submit(b"x", provider=None, fingerprint="fp-q-2")

    block.set()
    _wait_for_status(manager, job0, COMPLETED)


def test_get_unknown_job_id_returns_none(make_manager):
    manager = make_manager(max_concurrency=1)
    assert manager.get("does-not-exist") is None


def test_ttl_cleanup_removes_old_completed_jobs_from_get(make_manager, monkeypatch):
    monkeypatch.setattr(job_manager_module, "run_multi_agent_v1_analysis", lambda *a, **k: {"ok": True})
    monkeypatch.setenv(job_manager_module.JOB_TTL_SECONDS_ENV_VAR, "3600")
    manager = make_manager(max_concurrency=1)
    job_id, _ = manager.submit(b"x", provider=None, fingerprint="fp-ttl")
    _wait_for_status(manager, job_id, COMPLETED)

    import datetime as dt
    record = manager.store.get(job_id)
    record.completed_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    manager.store.update(job_id, completed_at=record.completed_at)

    assert manager.get(job_id) is None  # TTL already expired -> cleaned up on next read
