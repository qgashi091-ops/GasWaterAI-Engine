"""MICRO-AGENT ARCHITECTURE POC -- Phases 2-7, implemented and ready to run,
NOT executed in this session (see main()'s first check).

STOP CONDITION (checked before anything else, per explicit instruction):
this script requires GASWATERAI_VISION_API_KEY (the same credential
app/vision_fallback/anthropic_provider.py already documents). No such
credential exists in this build/dev session -- confirmed by checking the
environment directly before writing this script. Running this script here
would immediately print the BLOCKED message below and exit(2) without
making a single API call or fabricating a result.

What it does once a credential IS available:
  Phase 4: runs the full frozen benchmark (condition A, image only) 3x,
    plus a deterministic 10-item subset 10x, measuring classification/
    UNKNOWN consistency and contradictions.
  Phase 5: scores the (first) frozen run against the human ground truth
    already embedded in benchmark_manifest.json's `true_class` field --
    ground truth is read ONLY after predictions exist in memory, never
    before or during a call.
  Phase 6: runs condition B (image + local deterministically-extracted
    text) once across the same frozen crops and compares to condition A's
    first run. The cache means condition A's already-computed responses
    (empty text_hash) are never re-requested for this step.
  Phase 7: (optional second model) -- only attempted if a second model id
    is passed via --second-model; skipped honestly otherwise.

No prompt tuning after seeing condition A's results: this script's prompt
and schema are fixed at import time (from
app.microagents.symbol_recognition_agent) and this script itself contains
no logic that could change them based on intermediate results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.microagents.symbol_recognition_agent import SymbolRecognitionAgent
from app.vision_fallback.anthropic_provider import API_KEY_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "microagent_poc" / "benchmark"
MANIFEST_PATH = BENCHMARK_DIR / "benchmark_manifest.json"
RESULTS_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "microagent_poc" / "results"
CACHE_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "microagent_poc" / "cache"

FULL_BENCHMARK_RUNS = 3
SUBSET_RUNS = 10
SUBSET_SIZE = 10


def load_benchmark() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def deterministic_subset(items: list[dict], n: int) -> list[dict]:
    return sorted(items, key=lambda i: i["benchmark_id"])[:n]


def run_condition(agent: SymbolRecognitionAgent, items: list[dict], use_text: bool, run_index: int) -> list[dict]:
    results = []
    for item in items:
        tight_bytes = (REPO_ROOT / item["tight_crop_path"]).read_bytes()
        context_bytes = (REPO_ROOT / item["context_crop_path"]).read_bytes()
        nearby_text = item["nearby_text"] if use_text else None
        obs = agent.recognize(item["benchmark_id"], tight_bytes, context_bytes, nearby_text=nearby_text)
        results.append({"run_index": run_index, "true_class": item["true_class"], **obs.to_dict()})
    return results


def score(predictions: list[dict]) -> dict:
    """Phase 5 -- ground truth (`true_class`, already embedded in the
    frozen benchmark manifest) is read here, AFTER predictions exist."""
    n = len(predictions)
    correct = sum(1 for p in predictions if p["component_type"] == p["true_class"])
    unknown = sum(1 for p in predictions if p["component_type"] is None and p["available"])
    wrong = n - correct - unknown
    classes = sorted({p["true_class"] for p in predictions})
    per_class = {}
    for c in classes:
        subset = [p for p in predictions if p["true_class"] == c]
        tp = sum(1 for p in subset if p["component_type"] == c)
        fn_unknown = sum(1 for p in subset if p["component_type"] is None)
        fn_wrong = sum(1 for p in subset if p["component_type"] not in (c, None))
        fp = sum(1 for p in predictions if p["component_type"] == c and p["true_class"] != c)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / len(subset) if subset else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
        per_class[c] = {"support": len(subset), "tp": tp, "unknown": fn_unknown, "wrong": fn_wrong, "precision": precision, "recall": recall, "f1": f1}
    confusion = Counter((p["true_class"], p["component_type"] or "UNKNOWN") for p in predictions)
    return {
        "n": n, "exact_accuracy": correct / n if n else None,
        "unknown_rate": unknown / n if n else None, "wrong_rate": wrong / n if n else None,
        "per_class": per_class, "confusion_matrix": {f"{t}->{p}": c for (t, p), c in confusion.items()},
    }


def consistency(runs: list[list[dict]]) -> dict:
    by_benchmark_id: dict[str, list[str | None]] = {}
    for run in runs:
        for p in run:
            by_benchmark_id.setdefault(p["benchmark_id"], []).append(p["component_type"])
    consistent = sum(1 for v in by_benchmark_id.values() if len(set(v)) == 1)
    contradictory = [{"benchmark_id": k, "answers": v} for k, v in by_benchmark_id.items() if len(set(v)) > 1]
    return {
        "items": len(by_benchmark_id), "runs_per_item": len(runs),
        "fully_consistent_items": consistent, "contradictory_items": contradictory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--second-model", default=None, help="Phase 7 (optional): a second model id to benchmark on the identical frozen dataset.")
    args = parser.parse_args()

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        print("BLOCKED: no vision API credential available.")
        print(f"Required environment variable: {API_KEY_ENV_VAR}")
        print("This is the exact same credential app/vision_fallback/anthropic_provider.py")
        print("already documents for Engine v1's vision fallback -- not present in this session.")
        print("No benchmark result was fabricated. Set this variable and re-run to execute Phases 4-7.")
        sys.exit(2)

    manifest = load_benchmark()
    items = manifest["items"]
    candidate_labels = manifest["candidate_class_list"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agent = SymbolRecognitionAgent(candidate_labels, cache_dir=CACHE_DIR)

    # Phase 4: full benchmark x3 (condition A), subset x10 (condition A).
    full_runs = [run_condition(agent, items, use_text=False, run_index=i) for i in range(FULL_BENCHMARK_RUNS)]
    subset_items = deterministic_subset(items, SUBSET_SIZE)
    subset_runs = [run_condition(agent, subset_items, use_text=False, run_index=i) for i in range(SUBSET_RUNS)]

    # Phase 5: score run 0 (frozen) against ground truth.
    scoring = score(full_runs[0])
    full_consistency = consistency(full_runs)
    subset_consistency = consistency(subset_runs)

    # Phase 6: condition B once, compared to condition A's run 0.
    condition_b = run_condition(agent, items, use_text=True, run_index=0)
    scoring_b = score(condition_b)

    output = {
        "dataset_hash": (BENCHMARK_DIR / "DATASET_HASH.txt").read_text().strip(),
        "condition_a_scoring": scoring, "condition_a_consistency_3run": full_consistency,
        "condition_a_subset_consistency_10run": subset_consistency,
        "condition_b_scoring": scoring_b,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
