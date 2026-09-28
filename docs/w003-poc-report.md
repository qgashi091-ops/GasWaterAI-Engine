# W-003 PoC Report

## Input

`tests/fixtures/W-003_Referenzfall.Plan.pdf` (244,069 bytes, 1 page, real
native-text PDF export — no OCR fallback triggered) — the actual reference
plan behind the previously-observed Base44 defect (`Dead-End`/`Dead-End`/`Loop`
oscillation on an identical crop). Committed as a fixture, not synthesized.

## Extraction results

| Metric | Value |
|---|---|
| Pages | 1 |
| Page size | 3024 × 2160 pt |
| Text source | native |
| Graph nodes | 363 |
| Graph edges | 246 |
| Symbol candidates | 76 |
| Bridged junction gaps | 2 collinear, 4 symbol-junction, 3 corner-convergence (9 total) |
| Background line segments excluded | 4,211 of 4,459 candidate segments (94%) — one dominant color group correctly identified as a background/reference layer, not the pipe network |

## PlanFacts results

| Metric | Value |
|---|---|
| Total facts | 1,468 |
| `FACT` | 258 |
| `DERIVED_FACT` | 1,158 |
| `UNRESOLVED` | 52 |
| Connected components | 141 |
| — with a proven cycle | 12 |
| — acyclic | 129 |

Every single one of the 1,468 facts carries `fact_id`, `source`,
`evidence_status`, and `supporting_edges`/`supporting_nodes` — see
`examples/w003_planfacts.json` for the full, real response.

## The garden-valve topology probe

Located by searching extracted text for the vocabulary `Gartenventil`
(garden valve) — **not** by hardcoded coordinates or filenames; see
`scripts/w003_topology_probe.py`, which generalizes to any plan. This plan
has five garden-valve labels:

| Label | Associated to | `cycle_membership` | `dead_end_path` |
|---|---|---|---|
| "Gartenventil Nussbaum" | edge e61 | `False` | `PHYSICAL_END` (proven dead-end) |
| "Gartenventil Nussbaum" | symbol sym32 → edge e64 | `False` | `DEVICE_BOUNDARY` (proven dead-end) |
| "Gartenventil Nussbaum" | edge e66 | `False` | `PHYSICAL_END` (proven dead-end) |
| "Gartenventil Nussbaum" | edge e68 | `False` | `PHYSICAL_END` (proven dead-end) |
| "Gartenventil" | edge e78 | **`True`** | — (a cycle member has no dead-end path) |

**Finding:** four of the five real garden-valve connections in this plan are
graph-provably dead-ends; the fifth is graph-provably part of a closed,
directly-drawn cycle. None came back `UNRESOLVED`. The available vector graph
is sufficient to answer this question deterministically for every garden
valve on this specific plan — none required a guess.

The fifth case (a genuine loop member) is the most plausible candidate for
the historically-oscillating Base44 case: an LLM asked "loop or dead-end?"
about a location that is *actually* a loop is exactly the kind of case where
free-form judgment would inconsistently land on the right answer sometimes
and the wrong one other times — unlike the other four, which are
unambiguous, simple dead-ends that a model would have much less reason to
misclassify as a loop. This engine does not need to identify *which specific*
historical LLM run corresponds to which label — it deterministically proves
the topology for all five, which is the actual fix.

## Reproducibility (primary acceptance criterion)

`tests/test_reproducibility.py::test_w003_ten_repeated_runs_produce_semantically_identical_plan_facts`:
the real W-003 PDF was analyzed **10 times** in a single test run, each time
independently re-running the full pipeline (vector extraction → graph →
bridging → symbols → text → association → PlanFacts) from the same PDF
bytes, and each run's `plan_facts` block canonicalized (sorted keys, no
timing fields — timing lives only in the API's `diagnostics`, never inside
`plan_facts`) and SHA-256 hashed.

**Result: 10/10 runs produced byte-identical canonical `plan_facts` (a single
distinct hash across all 10 runs, 1,468 facts every time).**

```
$ pytest tests/test_reproducibility.py -v
test_w003_document_fingerprint_is_stable PASSED
test_w003_ten_repeated_runs_produce_semantically_identical_plan_facts PASSED
```

## Performance (real W-003, `scripts/measure_performance.py`, 5 runs)

| Metric | Value |
|---|---|
| Parse time (`pipeline.analyze_pdf_bytes`) | min 1.21s / avg 1.29s / max 1.42s |
| PlanFacts time (`plan_facts.build_document_facts`) | min 0.016s / avg 0.020s / max 0.029s |
| Peak process RSS (cumulative, 5 runs, one process) | ~186 MiB |

PlanFacts computation itself is negligible (~20ms — pure graph traversal,
O(V+E) over 363 nodes / 246 edges); essentially all wall-clock time is spent
in the *reused, unmodified* PyMuPDF-based vector/text extraction, exactly as
in the original repository. 186 MiB peak RSS for a single native-text page
(no OCR) is comfortably compatible with a modest cloud service; a page
requiring the tiled-OCR fallback would cost more (inherited from the
original parser, already designed around a tiling strategy specifically to
bound that cost — see `text.py`'s own module docstring).

## Tests

16/16 pass (`pytest tests/ -v`): 12 synthetic `plan_facts.py` unit tests
(chain, closed cycle, branch-off-a-cycle, bridge-required-closure staying
unresolved, mixed proven/unresolved endpoints, input-order independence,
coordinate-jitter independence, stable content-derived ids, self-loop
handling, the FACT/DERIVED_FACT/UNRESOLVED invariant, direct-association-only
dimension evidence), the 2 reproducibility tests above, and 3 live API tests
against the real W-003 fixture (health check, non-PDF rejection, full
`/analyze` response shape).

## Limitations

- Single-page PDF tested; multi-page cross-sheet topology is explicitly out
  of scope for v0.1 (per the task) and untested here.
- Dimension/text evidence covers only directly-associated labels — the more
  sophisticated symbol-port propagation-with-ambiguity-handling in the
  Base44 application's `vektorGraphMapper.ts` is not reimplemented.
- No scale calibration exists anywhere in this pipeline (inherited from the
  original parser) — 4×ID and any real-world-length rule remain unanswerable
  from this engine alone.
- Only one real reference plan (W-003) was exercised end-to-end in this PoC;
  the original repository's own test history across W-001…W-010 gives
  reasonable confidence the underlying parser modules generalize, but this
  PoC itself did not re-run all ten.
- Symbol/apparatus *type* classification is not attempted here (inherited
  scope boundary from the original parser, deferred to Base44's symbol
  library by design).

## Is this technically stronger than the current Base44 topology path?

Yes, for the specific class of question this PoC targets. The Base44
application's Visual-First/Lupendetail path asks a multimodal LLM to judge
loop-vs-dead-end from a raster crop, with no seed/temperature control through
the current `InvokeLLM` interface — proven non-reproducible on this exact
plan. This engine answers the identical question from the same underlying
vector geometry with an algorithm that is, by construction, a pure function
of its input: **10/10 identical outputs across repeated runs**, with
explicit, inspectable provenance on every claim, and an honest `UNRESOLVED`
answer (52 of 1,468 facts here) wherever the geometry genuinely does not
prove a claim — rather than a confident guess. It does not yet replace
everything Base44's LLM path does (component identity, professional-rule
interpretation, apparatus recognition remain genuinely LLM-dependent
problems, as the prior architecture review concluded) — but for
cycle-vs-dead-end specifically, on real data, it is unambiguously stronger.

## Verdict

**POC SUCCESS — CONTINUE EXTERNAL ENGINE**
