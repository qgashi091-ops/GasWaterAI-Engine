"""Unit tests for the POC InMemoryJobStore (job-mode epic). No HTTP, no
JobManager -- just the storage seam itself: create/get/update, the
fingerprint-based idempotency lookup, and TTL-based cleanup."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.multi_agent_v1.job_store import COMPLETED, FAILED, PROCESSING, QUEUED, InMemoryJobStore, JobRecord


def _now():
    return datetime.now(timezone.utc)


def test_create_and_get_round_trips():
    store = InMemoryJobStore()
    record = JobRecord(job_id="j1", fingerprint="fp1", status=QUEUED, created_at=_now())
    store.create(record)
    fetched = store.get("j1")
    assert fetched is not None
    assert fetched.job_id == "j1"
    assert fetched.status == QUEUED


def test_get_unknown_job_returns_none():
    store = InMemoryJobStore()
    assert store.get("does-not-exist") is None


def test_get_returns_a_copy_not_the_live_record():
    store = InMemoryJobStore()
    store.create(JobRecord(job_id="j1", fingerprint="fp1", status=QUEUED, created_at=_now()))
    fetched = store.get("j1")
    fetched.status = "mutated-by-caller"
    assert store.get("j1").status == QUEUED


def test_update_changes_fields_in_place():
    store = InMemoryJobStore()
    store.create(JobRecord(job_id="j1", fingerprint="fp1", status=QUEUED, created_at=_now()))
    store.update("j1", status=PROCESSING, started_at=_now())
    assert store.get("j1").status == PROCESSING
    store.update("j1", status=COMPLETED, result={"ok": True}, completed_at=_now())
    fetched = store.get("j1")
    assert fetched.status == COMPLETED
    assert fetched.result == {"ok": True}


def test_update_on_unknown_job_is_a_safe_no_op():
    store = InMemoryJobStore()
    store.update("does-not-exist", status=COMPLETED)  # must not raise


def test_find_active_by_fingerprint_finds_queued_and_processing():
    store = InMemoryJobStore()
    store.create(JobRecord(job_id="j1", fingerprint="fp-shared", status=QUEUED, created_at=_now()))
    found = store.find_active_by_fingerprint("fp-shared")
    assert found is not None and found.job_id == "j1"

    store.update("j1", status=PROCESSING)
    found = store.find_active_by_fingerprint("fp-shared")
    assert found is not None and found.job_id == "j1"


def test_find_active_by_fingerprint_ignores_terminal_jobs():
    store = InMemoryJobStore()
    store.create(JobRecord(job_id="j1", fingerprint="fp-done", status=QUEUED, created_at=_now()))
    store.update("j1", status=COMPLETED, result={}, completed_at=_now())
    assert store.find_active_by_fingerprint("fp-done") is None


def test_find_active_by_fingerprint_ignores_failed_jobs():
    store = InMemoryJobStore()
    store.create(JobRecord(job_id="j1", fingerprint="fp-failed", status=QUEUED, created_at=_now()))
    store.update("j1", status=FAILED, error="boom", completed_at=_now())
    assert store.find_active_by_fingerprint("fp-failed") is None


def test_find_active_by_fingerprint_no_match_returns_none():
    store = InMemoryJobStore()
    store.create(JobRecord(job_id="j1", fingerprint="fp-a", status=QUEUED, created_at=_now()))
    assert store.find_active_by_fingerprint("fp-b") is None


def test_delete_completed_before_removes_old_terminal_jobs_only():
    store = InMemoryJobStore()
    old_completed_at = _now() - timedelta(hours=2)
    store.create(JobRecord(job_id="old", fingerprint="fp1", status=QUEUED, created_at=old_completed_at))
    store.update("old", status=COMPLETED, result={}, completed_at=old_completed_at)

    recent_completed_at = _now()
    store.create(JobRecord(job_id="recent", fingerprint="fp2", status=QUEUED, created_at=recent_completed_at))
    store.update("recent", status=COMPLETED, result={}, completed_at=recent_completed_at)

    cutoff = _now() - timedelta(hours=1)
    deleted = store.delete_completed_before(cutoff)

    assert deleted == 1
    assert store.get("old") is None
    assert store.get("recent") is not None


def test_delete_completed_before_never_removes_active_jobs():
    store = InMemoryJobStore()
    old_created_at = _now() - timedelta(hours=2)
    store.create(JobRecord(job_id="still-running", fingerprint="fp1", status=PROCESSING, created_at=old_created_at))

    cutoff = _now() - timedelta(hours=1)
    deleted = store.delete_completed_before(cutoff)

    assert deleted == 0
    assert store.get("still-running") is not None
