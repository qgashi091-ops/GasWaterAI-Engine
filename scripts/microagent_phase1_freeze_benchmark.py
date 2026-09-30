"""MICRO-AGENT ARCHITECTURE POC -- Phase 1: freeze the symbol-recognition
benchmark, BEFORE any agent is run.

Ground truth rule (same discipline as the earlier Detector POC): only
`class` from the live `annotations` collection with
`verification_status == LABELED` counts. `NOT_A_COMPONENT` and `AMBIGUOUS`
are excluded entirely -- never truth, never in this benchmark. No
automatically-proposed label (`engine_suggestions`) is ever used as truth
or included in what the agent sees.

`OTHER_RELEVANT_SYMBOL` is not itself a usable class name (it is a human
catch-all bucket, not a component type -- see
docs/v04-detector-poc-phase0-report.md's breakdown). Where the annotator
left a free-text note naming the real component (Verteilbatterie,
Waschmaschine, ...), that note IS the human-confirmed label and is used as
this instance's true_class (one known typo normalized: "Badewannenmischrf"
-> "Badewannenmischer", confirmed unambiguous by comparing against the
sibling instance's own note). An instance with no note, or whose note names
a label/annotation rather than a component ("Beschriftung Verteilung"), is
excluded -- there is no human-confirmed component identity to freeze.

Only the 20 DEV-xx plans are read (data/dev_plans_v04/raw/); W-001..W-010
are never opened by this script.

For each selected candidate:
  - a TIGHT crop (just the candidate's own bbox) and a CONTEXT crop (the
    already-saved crop_path PNG, which already carries the v0.1-generated
    context margin) are both derived from the single already-rendered PNG
    -- no PDF re-rendering needed for the crops themselves.
  - nearby text is extracted DETERMINISTICALLY by re-opening the source
    plan PDF and reusing `app.plan_analysis.text.extract_native_text_spans`
    unmodified (the same frozen text-extraction module used everywhere
    else in this engine), converted to DISPLAY space, filtered to a fixed
    80pt margin around the candidate's own bbox -- exactly the convention
    already established in `canonical_inventory.py`'s `_text_near_component`.

Selection is capped per class (deterministic, sorted by candidate_id) to
keep the benchmark within the requested ~30-50 range while maximizing class
and plan/style diversity -- diversity is a goal of THIS benchmark (testing
raw recognition breadth), unlike Detector v1's training split, which needed
cross-style generalization instead.

Writes:
  data/dev_plans_v04/microagent_poc/benchmark/crops/<id>_tight.png
  data/dev_plans_v04/microagent_poc/benchmark/crops/<id>_context.png
  data/dev_plans_v04/microagent_poc/benchmark/benchmark_manifest.json
  data/dev_plans_v04/microagent_poc/benchmark/DATASET_HASH.txt
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
from PIL import Image

from app.plan_analysis.text import extract_native_text_spans

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"
RAW_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "raw"
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "microagent_poc" / "benchmark"
OUT_CROPS_DIR = OUT_DIR / "crops"

TEXT_MARGIN_PT = 80.0
CAP_PER_CLASS = 7  # keeps the total near the requested ~30-50 while giving every class a fair, bounded share

LABEL_TYPO_FIX = {"badewannenmischrf": "badewannenmischer"}
NOT_A_COMPONENT_NOTE = "beschriftung verteilung"  # a label, not a component -- excluded (see docstring)


def _raw_true_label(doc: dict) -> str | None:
    """The class/note string before cross-annotator casing is normalized --
    see `_true_class` for the canonicalization step (grouping
    "Badewannenmischer" and "badewannenmischer" as the same class)."""
    cls = doc["class"]
    if cls in ("NOT_A_COMPONENT", "AMBIGUOUS"):
        return None
    if cls != "OTHER_RELEVANT_SYMBOL":
        return cls
    note = (doc.get("annotator_note") or "").strip()
    if not note:
        return None
    key = note.casefold()
    if key == NOT_A_COMPONENT_NOTE:
        return None
    return LABEL_TYPO_FIX.get(key, note)


def _to_display(bbox: tuple, rotation_matrix) -> tuple:
    if rotation_matrix is None:
        return bbox
    r = pymupdf.Rect(*bbox) * rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)


def _nearby_text_for_candidate(plan_pdf_cache: dict, plan_id: str, page_number: int, bbox_page_space: tuple) -> list[str]:
    key = (plan_id, page_number)
    if key not in plan_pdf_cache:
        pdf_path = RAW_DIR / f"{plan_id}.pdf"
        pdf = pymupdf.open(str(pdf_path))
        page = pdf[page_number - 1]
        rotation_matrix = page.rotation_matrix if page.rotation else None
        spans = [
            (_to_display(s.bbox, rotation_matrix), s.text)
            for s in extract_native_text_spans(page) if s.text.strip()
        ]
        plan_pdf_cache[key] = spans
        pdf.close()
    spans = plan_pdf_cache[key]

    x0, y0, x1, y1 = bbox_page_space
    zone = (x0 - TEXT_MARGIN_PT, y0 - TEXT_MARGIN_PT, x1 + TEXT_MARGIN_PT, y1 + TEXT_MARGIN_PT)
    out = []
    for (sx0, sy0, sx1, sy1), text in spans:
        if sx0 < zone[2] and zone[0] < sx1 and sy0 < zone[3] and zone[1] < sy1:
            out.append(text)
    return out


def load_labels(dump_dir: Path) -> dict[str, dict]:
    out = {}
    for f in (dump_dir / "annotations" / "annotations").glob("*.json"):
        out[f.stem] = json.loads(f.read_text())
    return out


def main(dump_dir: Path) -> None:
    manifest = {c["candidate_id"]: c for c in json.loads(MANIFEST_PATH.read_text())}
    labels = load_labels(dump_dir)

    raw_pool: dict[str, list[dict]] = {}
    for cid, doc in sorted(labels.items()):
        if doc.get("verification_status") != "LABELED":
            continue
        raw_label = _raw_true_label(doc)
        if raw_label is None:
            continue
        cand = manifest.get(cid)
        if cand is None:
            continue
        raw_pool.setdefault(raw_label, []).append({"candidate_id": cid, **cand})

    # Canonicalize case-only variants of the same class (e.g.
    # "Badewannenmischer" vs "badewannenmischer", both real annotator notes)
    # into one class: group by casefold, use the lexicographically-first
    # literal variant as the display name -- deterministic, no capitalization
    # heuristic needed.
    pool: dict[str, list[dict]] = {}
    groups: dict[str, list[str]] = {}
    for raw_label in raw_pool:
        groups.setdefault(raw_label.casefold(), []).append(raw_label)
    for casefold_key, variants in groups.items():
        canonical_name = sorted(variants)[0]
        for variant in variants:
            pool.setdefault(canonical_name, []).extend(raw_pool[variant])

    selected: list[dict] = []
    for true_class in sorted(pool):
        entries = sorted(pool[true_class], key=lambda e: e["candidate_id"])[:CAP_PER_CLASS]
        for e in entries:
            selected.append({**e, "true_class": true_class})

    OUT_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    plan_pdf_cache: dict = {}
    benchmark_items = []
    for entry in selected:
        cid = entry["candidate_id"]
        context_png_path = Path(entry["crop_path"])
        context_img = Image.open(context_png_path).convert("RGB")

        # bbox_in_crop_px is in the SAVED PNG's own pixel grid (300 DPI,
        # see app/dataset_pipeline/candidates.py's own documented contract)
        bx0, by0, bx1, by1 = entry["bbox_in_crop_px"]
        pad = 12.0  # small fixed pixel pad so the tight crop isn't pixel-exact-zero margin
        tight_box = (
            max(0, bx0 - pad), max(0, by0 - pad),
            min(context_img.width, bx1 + pad), min(context_img.height, by1 + pad),
        )
        tight_img = context_img.crop(tuple(round(v) for v in tight_box))

        tight_path = OUT_CROPS_DIR / f"{cid}_tight.png"
        context_path = OUT_CROPS_DIR / f"{cid}_context.png"
        tight_img.save(tight_path)
        context_img.save(context_path)

        nearby_text = _nearby_text_for_candidate(
            plan_pdf_cache, entry["plan_id_pseudonymous"], entry["page"], tuple(entry["bbox_page_space"]),
        )

        benchmark_items.append({
            "benchmark_id": cid,
            "candidate_id": cid,
            "true_class": entry["true_class"],
            "plan_id": entry["plan_id_pseudonymous"],
            "style_family": entry["style_family"],
            "page": entry["page"],
            "tight_crop_path": str(tight_path.relative_to(REPO_ROOT)),
            "context_crop_path": str(context_path.relative_to(REPO_ROOT)),
            "tight_crop_sha256": hashlib.sha256(tight_path.read_bytes()).hexdigest(),
            "context_crop_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            "nearby_text": sorted(set(nearby_text)),
        })

    candidate_class_list = sorted({item["true_class"] for item in benchmark_items})

    manifest_out = {
        "candidate_class_list": candidate_class_list,
        "item_count": len(benchmark_items),
        "class_counts": {c: sum(1 for i in benchmark_items if i["true_class"] == c) for c in candidate_class_list},
        "plan_counts": {p: sum(1 for i in benchmark_items if i["plan_id"] == p) for p in sorted({i["plan_id"] for i in benchmark_items})},
        "style_family_counts": {f: sum(1 for i in benchmark_items if i["style_family"] == f) for f in sorted({i["style_family"] for i in benchmark_items})},
        "items": benchmark_items,
    }

    manifest_path = OUT_DIR / "benchmark_manifest.json"
    manifest_json = json.dumps(manifest_out, indent=2, ensure_ascii=False, sort_keys=True)
    manifest_path.write_text(manifest_json)

    dataset_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    (OUT_DIR / "DATASET_HASH.txt").write_text(dataset_hash + "\n")

    print(f"Selected {len(benchmark_items)} benchmark items across {len(candidate_class_list)} classes")
    print(json.dumps(manifest_out["class_counts"], indent=2, ensure_ascii=False))
    print(f"Plans: {manifest_out['plan_counts']}")
    print(f"Style families: {manifest_out['style_family_counts']}")
    print(f"Dataset hash (SHA-256 of the sorted-key manifest JSON): {dataset_hash}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.dump_dir)
