# Micro-Agent Architecture POC — Report

Status: **Phases 1–3 implemented and validated. Phases 4–7 blocked before
any evaluation began** — no vision API credential exists in this
build/dev session. Per explicit instruction, no benchmark result was
fabricated; this report documents exactly what is ready and exactly what
is missing to run it.

## Credential check (done first, per instruction)

`app/vision_fallback/anthropic_provider.py` (built in the Engine v1 epic)
requires `GASWATERAI_VISION_API_KEY`. Checked directly in this environment
before writing any benchmark code: not set, and no equivalent
(`ANTHROPIC_API_KEY` or similar) is present either. `scripts/microagent_phase2to7_run_benchmark.py`
enforces this as its first action — running it now prints:

```
BLOCKED: no vision API credential available.
Required environment variable: GASWATERAI_VISION_API_KEY
```

and exits before making a single API call. **This is the exact credential
required to run Phases 4–7.**

## 1. Benchmark dataset composition (Phase 1 — frozen)

`scripts/microagent_phase1_freeze_benchmark.py` selected **47 human-
confirmed positive component crops** across **14 distinct component
types**, **7 plans**, **4 style families** — from the live `annotations`
collection (`verification_status == LABELED`), excluding `NOT_A_COMPONENT`
and `AMBIGUOUS` entirely, and never using `engine_suggestions` (automatic
labels) as truth. `OTHER_RELEVANT_SYMBOL` instances were resolved to their
annotator's own free-text note as the true class (the same treatment used
in the earlier Detector POC's Phase 0 audit), excluding notes with no
component name and the one note that named a label rather than a
component ("Beschriftung Verteilung"). A genuine cross-annotator casing
variant ("Badewannenmischer" vs "badewannenmischer") was merged into one
class deterministically (lexicographically-first spelling), not by a
capitalization heuristic.

| Class | Count |
|---|---|
| küche | 7 |
| wc up | 7 |
| dusche | 7 |
| up verteiler unter wt | 7 |
| Verteilbatterie | 3 |
| Waschmaschine | 3 |
| Badewannenmischer | 3 |
| Zirkulationsventile | 3 |
| Gartenventil Frostsicher | 2 |
| Sicherheitsventil | 1 |
| Duschenmischer | 1 |
| Waschtrog | 1 |
| Waschtisch | 1 |
| secomat | 1 |

Plans: DEV-01 (3), DEV-03 (19), DEV-04 (10), DEV-05 (12), DEV-06 (1),
DEV-17 (1), DEV-20 (1). Style families: FAM-4 (23), STANDALONE-3 (19),
STANDALONE-1 (3), STANDALONE-14 (1), STANDALONE-17 (1). W-001..W-010 were
never opened by this script.

**Dataset hash** (SHA-256 of the sorted-key benchmark manifest JSON):
see `data/dev_plans_v04/microagent_poc/benchmark/DATASET_HASH.txt`
(committed). Each crop's own SHA-256 is stored per-item in
`benchmark_manifest.json` for tamper-evident reproducibility.

For every item: a TIGHT crop (just the component) and a CONTEXT crop (the
already-rendered wider region, reused from the existing v0.1 candidate
pipeline output — no new PDF rendering needed for this), plus nearby text
extracted deterministically by re-opening the source plan PDF and reusing
`app.plan_analysis.text.extract_native_text_spans` unmodified (the same
frozen module used everywhere else in this engine), within an 80pt margin.

## 2. Model / provider / configuration (as configured, not yet executed)

- Provider: direct Anthropic Messages API (`app/vision_fallback/anthropic_provider.py`'s
  constants — same credential/endpoint as Engine v1's vision fallback, NOT
  Base44 InvokeLLM).
- Model: `claude-sonnet-5-5` (configurable via `GASWATERAI_VISION_MODEL`,
  recorded per-call).
- Temperature: `0` (fixed, for the stability question Phase 4 asks).
- Output: forced tool use (`classify_symbol`), strict JSON schema exactly
  matching the requested contract (`component_type`, `evidence`,
  `ambiguity`, `confidence`).

## 3. Exact micro-agent prompt size

- System prompt: **414 characters / 71 words** (see
  `app/microagents/symbol_recognition_agent.py`'s `SYSTEM_PROMPT` —
  reproduced in full below). No chain-of-thought request, no professional
  rule text, no plan-wide interpretation.
- Tool schema (JSON): 731 characters (for a 3-label example; scales with
  the candidate-class-list length only).
- Per-call user content: 1–2 base64 images (tight + optional context crop)
  + one short text block naming the candidate labels and, in condition B
  only, the deterministically-extracted nearby text.

> "You identify a single plumbing/sanitary component shown in an image
> crop from a technical plan. Choose exactly one label from the given
> candidate list, or UNKNOWN if the crop does not clearly show one of
> them. Base your answer only on what is visually present. A cautious
> UNKNOWN is better than a confident wrong guess -- never force a
> classification you are not sure of. Respond only via the
> classify_symbol tool."

## 4–11. IMAGE ONLY / IMAGE+TEXT results, per-class performance, UNKNOWN
rate, 3-run and 10-run consistency, contradictions, API-call count/cost

**Not available — BLOCKED at the credential gate before any call was
made.** `scripts/microagent_phase2to7_run_benchmark.py` implements all of
this (scoring, consistency, confusion matrix, condition A/B comparison,
cache-based call minimization) and is ready to run the moment
`GASWATERAI_VISION_API_KEY` is set — see that script for the exact,
frozen logic that would produce these numbers. No number is reported here
because none was measured.

## 12. Comparison to Detector v1 and handcrafted approaches

Cannot be made quantitatively without a real run. For context, both prior
approaches produced clean, measured negative or inconclusive results
against real plan data: Detector v1 (`docs/v1-engine-product-slice-report.md`)
measured 0% held-out cross-style recall for both its trained classes;
every handcrafted symbol-matching attempt before it
(`docs/v04-legend-structural-benchmark-report.md`,
`data/dev_plans_v04/symbol_reid_experiment/report.md`) also failed. This
POC's frozen benchmark (Phase 1) and agent implementation (Phase 2/3) are
ready to produce the first real comparison point the moment a credential
exists.

## 13. Verdict

**None of the three required verdict strings can be honestly issued.**
Each presupposes a measured result (`PROMISING`, `PARTIAL`, or `FAILED`
all describe an outcome of running the benchmark), and Phases 4–7 never
ran. Per the explicit instruction not to fabricate a benchmark result,
this POC's status is:

**Phases 1–3: DONE, validated (15 automated tests: 6 for the underlying
vision-fallback interface, 9 new for `SymbolRecognitionAgent`, all
fixture-based, no live call). Phases 4–7: BLOCKED, credential required:
`GASWATERAI_VISION_API_KEY`.**

Because Phases 1–7 did not pass, the conditional Phase 8 work (minimal
interfaces for `TextInterpretationAgent` and `RuleAssessmentAgent`) was
**not started**, per the task's own "IF AND ONLY IF" condition.

## What is needed to produce a real verdict

1. Set `GASWATERAI_VISION_API_KEY` (an Anthropic API key) in this
   environment.
2. Run `python scripts/microagent_phase2to7_run_benchmark.py`.
3. That single command executes Phases 4–7 exactly as specified (3x + 10x-
   subset repetition, scoring against the already-frozen ground truth,
   condition A vs. B, optional `--second-model`) and writes
   `data/dev_plans_v04/microagent_poc/results/results.json`.
4. Re-run this report's sections 4–13 from that file's actual numbers.

No code change is needed to produce a real result — only the credential.
