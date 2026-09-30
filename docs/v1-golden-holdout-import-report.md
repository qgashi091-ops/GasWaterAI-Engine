# Golden Holdout Import — W-001..W-010

Audit of the uploaded ZIP, current status of the 38-golden-finding
benchmark denominator, and the holdout-isolation protection now in place.
No optimization, tuning, or benchmark run was performed.

## 1. ZIP audit

| Plan | Present | Readable | Notes |
|---|---|---|---|
| W-001 | yes | yes | |
| W-002 | yes | yes | |
| W-003 | yes | yes | identical to the plan already committed at `tests/fixtures/W-003_Referenzfall.Plan.pdf` |
| W-004 | yes | yes | |
| W-005 | yes | yes | |
| W-006 | yes | yes | |
| W-007 | yes | yes | |
| W-008 | yes | yes | scanned/rasterized -- 0 native text spans, needs OCR |
| W-009 | yes | yes | |
| W-010 | yes | yes | **content-duplicate of W-002, see below** |

- **10/10 plan PDFs present**, all valid single-page PDFs (opened without
  error, `%PDF` header, readable by PyMuPDF).
- **0 Fachbericht files present.** The ZIP contains only
  `Fall <n>/W-0XX_Referenzfall.Plan.pdf` -- no Fachbericht PDF, DOCX, or any
  other document. "Plan ↔ Fachbericht" pairing cannot be verified against
  files in this ZIP because only one half of the pair (the plan) is here.
- **No byte-identical duplicates** among the 10 PDFs (distinct SHA-256 for
  all 10).
- **W-010 is a content-duplicate of W-002**, confirmed directly (not
  inferred from the coincidental matching page size, which W-008 also
  shares): identical extracted text (2,604 characters, same content) and
  identical vector drawing-item count (20,308) on both files. Different
  SHA-256 and a ~7-minute-later PDF creation timestamp indicate a
  re-export/re-save of the same source plan, not an independent 10th case.
  Recorded in `tests/golden_holdout/harness.py`'s `KNOWN_CONTENT_DUPLICATES`
  for whoever later curates the benchmark denominator to account for.

## 2. Golden ground truth (38 findings) — STOPPED, not fabricated

Per instruction, searched for an existing curated 38-finding mapping before
doing anything else with ground truth, across all three repositories
available in this session, including full `git log --all` history on every
branch of each:

- `qgashi091-ops/GasWaterAI-Engine` (this repo)
- `qgashi091-ops/gaswaterai`
- `qgashi091-ops/gaswaterai-base44-app` — this one has a properly engineered
  **Golden Test Harness** (`base44/shared/goldenTestHarness.ts` +
  `base44/shared/data/goldenGroundTruth/fachberichte.json`), with its own
  isolation-guard test (`goldenTestHarness.test.mjs`) already enforcing
  exactly the kind of protection requested in step 3 of this task, on the
  Base44 side.

**What that harness actually contains: 113 raw, itemized Fachbericht
bullets** (verbatim, PII-scrubbed-by-construction sentences from the 10
Fachbericht PDFs), 11–18 per plan, summing to exactly 113 — this is the
"113 Roh-Bullets" the task explicitly warned against treating as the
benchmark denominator, confirmed by direct count.

**The curated 38-golden-finding mapping itself does not exist as a
committed, machine-readable file anywhere.** One commit message in
`gaswaterai-base44-app`'s history even says so explicitly at the time:
"no machine-readable Golden ground truth existing yet." No `golden_findings_38.json`
or equivalent was found on any branch, in any of the three repositories.

**Per instruction: STOPPED here.** No new 38-finding denominator was
generated from the Fachbericht text or the 113 raw bullets. The 113 raw
bullets were imported as provenance only (see below), clearly labeled as
NOT the benchmark denominator, with a loader (`load_curated_benchmark()`)
that raises an explicit error rather than silently substituting them.

## 3. Holdout protection (implemented)

Mirrors the Base44 app's own isolation pattern for the same 10 plans:

- `tests/golden_holdout/plans/` — the 10 real plan PDFs, **gitignored**
  (same privacy category as `data/dev_plans_v04/raw/`: real client
  deliverables, never committed).
- `tests/golden_holdout/raw_findings_113.json` — the 113 raw bullets,
  copied from `gaswaterai-base44-app` with a `_provenance` block (source
  repo, path, commit hash) and an explicit "NOT THE BENCHMARK DENOMINATOR"
  status field. Committed (text-only, PII-scrubbed at extraction time per
  the source harness's own verified guarantee).
- `tests/golden_holdout/harness.py` — the only intended reader of the
  above; `load_curated_benchmark()` raises rather than ever falling back to
  the raw bullets.
- `tests/test_golden_holdout_isolation.py` — **4 automated tests**,
  passing: no `app/` module references the golden holdout; no script other
  than `scripts/run_golden_evaluation.py` references it; the plans
  directory is gitignored; no generator for the curated benchmark file
  exists in the repo.
- `scripts/run_golden_evaluation.py` updated to discover the now-present
  10 plans automatically, but its own `main()` still exits with `BLOCKED`
  before running the engine against them, because the curated
  `golden_findings_38.json` still does not exist — verified by reading the
  control flow, not executed against the real plans in this step (per
  instruction: no benchmark run yet).

Evaluation ordering discipline (ground truth read only after engine
output): encoded in `tests/golden_holdout/__init__.py`'s own contract and
already reflected in `run_golden_evaluation.py`'s existing structure
(`engine_outputs` is computed before any ground-truth file would ever be
read).

## Status report

| Item | Value |
|---|---|
| Anzahl Pläne | 10 / 10 |
| Anzahl Fachberichte (als Dateien im ZIP) | 0 |
| W-001..W-010 vollständig (Pläne) | ja |
| W-001..W-010 vollständig (Fachberichte) | nein -- keine im ZIP enthalten |
| Status 38 Golden Findings | **nicht vorhanden** in keinem der 3 Repos (nur 113 rohe Bullets, nicht der Benchmark-Nenner) |
| Fehlende Dateien/Daten | die 10 Fachbericht-Dokumente selbst; die kuratierte 38-Finding-Zuordnung |
| Holdout-Schutz | implementiert, 4/4 Isolations-Tests grün |
| Duplikate | keine byte-identischen; W-010 inhaltliches Duplikat von W-002 (bestätigt) |

No optimization, tuning, rule development, or benchmark run was performed.
