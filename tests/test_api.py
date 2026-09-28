from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import ENGINE_VERSION, app

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "engine_version": ENGINE_VERSION}


def test_analyze_rejects_non_pdf():
    r = client.post("/analyze", files={"file": ("not_a_pdf.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_analyze_real_w003_pdf_returns_the_required_shape():
    with FIXTURE.open("rb") as f:
        r = client.post("/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["engine_version"] == ENGINE_VERSION
    assert len(body["document_fingerprint"]) == 64
    assert isinstance(body["pages"], list) and len(body["pages"]) == 1
    assert "plan_facts" in body and "facts" in body["plan_facts"]
    assert all(f["kind"] in ("FACT", "DERIVED_FACT", "UNRESOLVED") for f in body["plan_facts"]["facts"])
    assert "diagnostics" in body and "timing_ms" in body["diagnostics"]
