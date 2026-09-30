"""Shared fixtures for the multi-agent v1 test suite. Builds small, REAL
synthetic PDFs (via pymupdf) and runs them through the existing, unmodified
deterministic pipeline (app.plan_analysis.pipeline.analyze_pdf_bytes,
app.plan_analysis.plan_facts.build_document_facts) so tests exercise real
schema.DocumentAnalysis/PlanFact objects rather than hand-faked ones.
"""
from __future__ import annotations

import pymupdf
import pytest

from app.plan_analysis import plan_facts as plan_facts_module
from app.plan_analysis.pipeline import analyze_pdf_bytes


def _build_pdf_bytes(page_builder) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=2000, height=2000)
    page_builder(page)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def legend_and_riser_pdf_bytes() -> bytes:
    """One page: a headed legend table (top) + a small drawn pipe riser with
    KW/WW labels, a safety-device label with an explicit category, a
    circulation-pump label, and a sampling-point label (bottom)."""

    def build(page: "pymupdf.Page") -> None:
        # Legend heading + several short rows, so legend_detection's
        # heading_match path fires (mirrors the real "Legende Sanitär" case
        # already found live in Detector v2's work).
        page.insert_text((100, 100), "Legende Sanitär", fontsize=14)
        for i, label in enumerate(["Wasserzaehler", "Absperrarmatur", "Filter", "Sicherheitsventil", "Systemtrenner"]):
            y = 130 + i * 20
            page.insert_text((100, y), label, fontsize=10)

        # A small drawn pipe segment (two collinear points -> one edge).
        shape = page.new_shape()
        shape.draw_line((300, 1000), (300, 1200))
        shape.finish(width=2)
        shape.commit()
        page.insert_text((320, 1050), "KW 1 LU", fontsize=10)
        page.insert_text((320, 1080), "WW 1 LU", fontsize=10)

        page.insert_text((600, 1000), "Sicherheitsventil Kategorie 2", fontsize=10)
        page.insert_text((600, 1100), "Zirkulationspumpe", fontsize=10)
        page.insert_text((600, 1200), "Probenahmestelle", fontsize=10)

    return _build_pdf_bytes(build)


@pytest.fixture
def heizband_pdf_bytes() -> bytes:
    def build(page: "pymupdf.Page") -> None:
        page.insert_text((300, 1000), "Heizband Begleitheizung", fontsize=10)

    return _build_pdf_bytes(build)


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    return _build_pdf_bytes(lambda page: None)


def analyze(pdf_bytes: bytes):
    doc = analyze_pdf_bytes(pdf_bytes, filename="test.pdf")
    facts = plan_facts_module.build_document_facts(doc)
    return doc, facts
