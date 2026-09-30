"""ISOLATION GUARD for the golden W-001..W-010 holdout -- a regression test,
not a one-time manual audit claim, mirroring the Base44 app's own
`goldenTestHarness.test.mjs` guard for the same 10 plans.

Fails the moment any production/engine code path (`app/`) or any
training/tuning script references the golden holdout package or its data
files, directly or by filename.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
SCRIPTS_DIR = REPO_ROOT / "scripts"

FORBIDDEN_PATTERN = re.compile(r"golden_holdout|golden_findings_38|raw_findings_113")

# The only legitimate references anywhere in the repo: the harness package
# itself, this test, and the dedicated evaluation script (which must run
# the frozen engine and obtain its output before it may call the harness --
# see golden_holdout/__init__.py's own ordering-discipline note; this test
# checks isolation, not call order, which cannot be verified statically).
ALLOWED_FILES = {
    REPO_ROOT / "scripts" / "run_golden_evaluation.py",
}


def _iter_python_files(directory: Path):
    if not directory.exists():
        return
    for path in directory.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_app_module_references_golden_holdout():
    hits = []
    for path in _iter_python_files(APP_DIR):
        if FORBIDDEN_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == [], f"app/ must never reference the golden holdout: {hits}"


def test_no_non_allowlisted_script_references_golden_holdout():
    hits = []
    for path in _iter_python_files(SCRIPTS_DIR):
        if path in ALLOWED_FILES:
            continue
        if FORBIDDEN_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == [], f"only scripts/run_golden_evaluation.py may reference the golden holdout: {hits}"


def test_golden_plans_directory_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "tests/golden_holdout/plans" in gitignore, (
        "the real golden plan PDFs must never be committed -- add "
        "tests/golden_holdout/plans/ to .gitignore"
    )


def test_curated_benchmark_not_fabricated():
    """Guards against ever silently generating golden_findings_38.json from
    the 113 raw bullets -- if this file appears, it must be a real,
    reviewed curation, never an automated derivation. This test simply
    documents that no such generator exists in this repo."""
    generators = []
    for path in _iter_python_files(REPO_ROOT):
        if "golden_holdout" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "golden_findings_38" in text and "write" in text.lower():
            generators.append(str(path.relative_to(REPO_ROOT)))
    assert generators == [], f"no script may auto-generate the curated benchmark file: {generators}"
