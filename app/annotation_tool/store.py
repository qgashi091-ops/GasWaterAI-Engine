"""Local, file-based annotation store for the v0.4 annotation tool.

No database server, no Base44, no network dependency -- a single JSON
manifest file IS the store, matching this whole engine's "no infrastructure
beyond what's already required" discipline. Every write is a full
rewrite-in-place of the manifest (small dataset, correctness over
cleverness) preceded by a write-to-temp-then-rename so a crash mid-save
never corrupts the file the product owner's annotation session depends on.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Optional

_LOCK = threading.Lock()


class AnnotationStore:
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        with open(manifest_path, encoding="utf-8") as f:
            self.records: list[dict] = json.load(f)
        self._by_id = {r["candidate_id"]: r for r in self.records}

    def _save(self) -> None:
        directory = os.path.dirname(self.manifest_path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".manifest_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.manifest_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def progress(self) -> dict:
        total = len(self.records)
        annotated = sum(1 for r in self.records if r["verification_status"] in ("LABELED", "SKIPPED"))
        by_status = {}
        for r in self.records:
            by_status[r["verification_status"]] = by_status.get(r["verification_status"], 0) + 1
        return {"annotated": annotated, "total": total, "by_status": by_status}

    def get(self, candidate_id: str) -> Optional[dict]:
        return self._by_id.get(candidate_id)

    def list_ids_in_order(self) -> list[str]:
        return [r["candidate_id"] for r in self.records]

    def next_unlabeled_id(self, after_id: Optional[str] = None) -> Optional[str]:
        ids = self.list_ids_in_order()
        start = 0
        if after_id and after_id in ids:
            start = ids.index(after_id) + 1
        for cid in ids[start:] + ids[:start]:
            if self._by_id[cid]["verification_status"] == "UNLABELED":
                return cid
        return None

    def annotate(
        self, candidate_id: str, *, assigned_class: Optional[str] = None,
        status: str = "LABELED", corrected_bbox_in_crop_px: Optional[list] = None,
        annotator_note: Optional[str] = None,
    ) -> dict:
        with _LOCK:
            rec = self._by_id.get(candidate_id)
            if rec is None:
                raise KeyError(candidate_id)
            rec["class"] = assigned_class
            rec["verification_status"] = status
            if corrected_bbox_in_crop_px is not None:
                rec["corrected_bbox_in_crop_px"] = corrected_bbox_in_crop_px
            if annotator_note is not None:
                rec["annotator_note"] = annotator_note
            self._save()
            return rec
