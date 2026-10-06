"""HTTP-level tests for the asynchronous job-mode pair
POST /multi_agent_v1/jobs + GET /multi_agent_v1/jobs/{job_id} (job-mode
epic): fast 202 on submit, the full queued -> processing -> completed
lifecycle, auth parity with the existing /multi_agent_v1/analyze
protection, 404 for an unknown job_id, idempotent duplicate submission,
and byte-identical results against the synchronous endpoint for the same
input. /analyze, /check and /multi_agent_v1/analyze themselves are
covered by their own existing test files and are not touched here."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import MULTI_AGENT_API_KEY_ENV_VAR, app
from app.multi_agent_v1.job_manager import reset_job_manager_for_tests
from app.multi_agent_v1.job_store import COMPLETED, FAILED, QUEUED

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_job_manager():
    """Every test gets its own empty in-memory job store and worker pool
    -- otherwise the fixture PDF's constant document_fingerprint would
    make one test's job look like an idempotent duplicate of another's."""
    reset_job_manager_for_tests()
    yield
    reset_job_manager_for_tests()


def _post_job(filename: str = None, headers: dict = None):
    with FIXTURE.open("rb") as f:
        return client.post(
            "/multi_agent_v1/jobs", files={"file": (filename or FIXTURE.name, f, "application/pdf")},
            headers=headers or {},
        )


def _poll_until_terminal(job_id: str, headers: dict = None, timeout: float = 240.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        r = client.get(f"/multi_agent_v1/jobs/{job_id}", headers=headers or {})
        assert r.status_code == 200
        last = r.json()
        if last["status"] in (COMPLETED, FAILED):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal status within {timeout}s; last={last}")


def test_submit_returns_202_immediately(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    t0 = time.monotonic()
    r = _post_job()
    elapsed = time.monotonic() - t0
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body and isinstance(body["job_id"], str) and body["job_id"]
    assert body["status"] in (QUEUED, "processing")
    assert elapsed < 5.0  # must not wait for the analysis itself


def test_job_lifecycle_reaches_completed_with_the_canonical_structure(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    job_id = _post_job().json()["job_id"]
    final = _poll_until_terminal(job_id)
    assert final["status"] == COMPLETED
    assert final["error"] is None
    assert "canonical_plan_understanding" in final["result"]
    assert final["result"]["canonical_plan_understanding"]["engine"] == "multi_agent_v1"
    assert final["created_at"] is not None
    assert final["started_at"] is not None
    assert final["completed_at"] is not None


def test_unknown_job_id_returns_404(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    r = client.get("/multi_agent_v1/jobs/does-not-exist")
    assert r.status_code == 404


def test_invalid_pdf_job_fails_with_a_safe_error(monkeypatch, tmp_path):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    not_a_pdf = tmp_path / "not_a_plan.pdf"
    not_a_pdf.write_bytes(b"this is not a pdf")
    with not_a_pdf.open("rb") as f:
        r = client.post("/multi_agent_v1/jobs", files={"file": (not_a_pdf.name, f, "application/pdf")})
    # Rejected synchronously -- never even enters the queue, same as /analyze.
    assert r.status_code == 400


def test_background_pipeline_failure_surfaces_as_a_failed_job(monkeypatch, tmp_path):
    """A file that passes the upfront %PDF check but fails deeper parsing
    (analyze_pdf_bytes itself raises) must not crash the worker thread or
    hang the job -- it must reach `failed` with a technical message."""
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nnot actually a valid pdf body at all")
    with broken.open("rb") as f:
        r = client.post("/multi_agent_v1/jobs", files={"file": (broken.name, f, "application/pdf")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    final = _poll_until_terminal(job_id)
    assert final["status"] == FAILED
    assert final["result"] is None
    assert final["error"]
    assert "PDF analysis failed" in final["error"]


def test_post_jobs_stays_open_when_key_not_configured(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    assert _post_job().status_code == 202


def test_post_jobs_rejects_missing_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    assert _post_job().status_code == 401


def test_post_jobs_rejects_wrong_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    assert _post_job(headers={"X-API-Key": "wrong-secret"}).status_code == 401


def test_post_jobs_accepts_correct_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    assert _post_job(headers={"X-API-Key": "the-real-secret"}).status_code == 202


def test_get_job_rejects_missing_key_once_configured(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    job_id = _post_job().json()["job_id"]
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    r = client.get(f"/multi_agent_v1/jobs/{job_id}")
    assert r.status_code == 401


def test_get_job_accepts_correct_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    job_id = _post_job(headers={"X-API-Key": "the-real-secret"}).json()["job_id"]
    r = client.get(f"/multi_agent_v1/jobs/{job_id}", headers={"X-API-Key": "the-real-secret"})
    assert r.status_code == 200


def test_duplicate_submission_of_identical_input_returns_the_same_job_id(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    r1 = _post_job()
    r2 = _post_job()
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["job_id"] == r2.json()["job_id"]


def test_analyze_check_and_jobs_endpoints_do_not_interfere(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    with FIXTURE.open("rb") as f:
        r_analyze = client.post("/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})
    with FIXTURE.open("rb") as f:
        r_check = client.post("/check", files={"file": (FIXTURE.name, f, "application/pdf")})
    r_job = _post_job()
    assert r_analyze.status_code == 200
    assert r_check.status_code == 200
    assert r_job.status_code == 202


def test_job_result_is_byte_identical_to_the_synchronous_endpoint(monkeypatch):
    """Section 3 of the job-mode spec: a successful job's result must
    equal POST /multi_agent_v1/analyze's result for identical input --
    both call the exact same run_multi_agent_v1_analysis() helper."""
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    with FIXTURE.open("rb") as f:
        sync_body = client.post("/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")}).json()

    job_id = _post_job().json()["job_id"]
    final = _poll_until_terminal(job_id)

    assert final["result"]["canonical_plan_understanding"] == sync_body["canonical_plan_understanding"]
    assert final["result"]["document_fingerprint"] == sync_body["document_fingerprint"]
    assert final["result"]["engine_version"] == sync_body["engine_version"]
