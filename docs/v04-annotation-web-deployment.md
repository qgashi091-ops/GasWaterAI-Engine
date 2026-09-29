# v0.4: hosted browser annotation tool (no local install required)

## Why this exists

The local FastAPI annotation tool (`app/annotation_tool/`) requires running
`uvicorn` on the reviewer's own machine. On a locked-down corporate laptop
where installing software is not possible, that tool cannot be used at all.
This deployment reuses the exact same 837 privacy-checked candidates and the
same `manifest.json`/`classes.json` produced by `scripts/run_v04_pipeline.py`,
without reprocessing any PDF and without generating a single new candidate,
and makes them reachable from an ordinary browser instead.

## What changed, and what didn't

- **Unchanged**: the annotation dataset itself (837 candidates from
  `data/dev_plans_v04/annotation_dataset/manifest.json`), the taxonomy
  (`classes.json`), and every privacy check that ran before a crop or a
  suggestion label was ever written to disk (see `docs/v04-dataset-report.md`).
  No PDF was re-parsed and no candidate was regenerated for this deployment.
- **New**: `scripts/build_annotation_web_export.py` reshapes the manifest into
  a slim, browser-friendly export (`data/dev_plans_v04/annotation_web_export/`)
  that drops fields meaningless or unsafe outside a local filesystem
  (`crop_path`, `bbox_page_space`, `crop_bbox_page_space`) and drops the
  mutable annotation fields (`class`, `verification_status`,
  `annotator_note`), which become per-viewer database rows in the hosted
  tool instead of static page content.
- **New**: the 837 crop PNGs (already-rendered, already privacy-checked
  images — the same files `app/annotation_tool` would serve locally) were
  uploaded once to a Claude Artifact's asset store and their resulting
  URLs stitched into a `candidates.json` used only by the hosted page. That
  stitching step is not committed to this repository (see below).

## Hosting mechanism

The hosted tool is a single-page Claude Artifact (a `claude.ai` page), not a
new FastAPI deployment:

- **No installation, ever**: the reviewer opens an HTTPS link in any
  browser. Nothing runs on their machine besides the browser tab.
- **Not publicly reachable**: an Artifact that declares the `assets`
  capability is organization-internal by platform guarantee — never a
  public URL — on top of which the artifact itself is private by default
  (shared only with people the owner explicitly grants access to).
- **An additional access code** (client-side SHA-256 check) gates the page
  itself, as a second, independent layer for this proof-of-concept, on top
  of the platform's own access control. The code is not stored in this
  repository; it was given to the requester directly.
- **No raw PDFs are published anywhere.** Only the pre-rendered, already
  privacy-checked crop PNGs were uploaded as assets. This means the local
  tool's "larger context" feature (a live re-render of a wider region from
  the raw PDF, itself gated by two independent PII scans — see
  `app/annotation_tool/main.py`) cannot be offered in the hosted version:
  there is no raw PDF for it to render from. The hosted page says so
  explicitly instead of silently omitting the feature.
- **Persistent storage**: every annotation (`class`, `verification_status`,
  `annotator_note`) is written to the Artifact's own server-side database
  (the `db` runtime capability) — not to the browser, not to any container
  filesystem. It survives page reloads, browser restarts, and the hosted
  page being republished with a new version.

## Not in scope for this deployment

Same boundaries as the rest of v0.4: no Base44 production integration, no
detector training, and no change to the annotation logic or taxonomy. This
is a delivery-mechanism change only.

## Files in this repository

- `scripts/build_annotation_web_export.py` — the export-shaping script
  (reusable; re-run it any time `manifest.json`/`classes.json` change).
- `data/dev_plans_v04/annotation_web_export/candidates.json` — 837 slim
  records, **without** any asset URL (that stitching happens only in the
  hosted Artifact's own copy, generated interactively once per deployment,
  and is not reproducible from this repo alone without re-uploading the
  crops).
- `data/dev_plans_v04/annotation_web_export/classes.json` — copy of the
  taxonomy used by the hosted page's class buttons.

Not committed here: the access-code hash, the crop PNGs themselves (already
gitignored under `data/dev_plans_v04/`), and the URL-stitched copy of
`candidates.json` that the hosted page actually serves.
