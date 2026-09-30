# ONE-PLAN SYMBOL REIDENTIFICATION EXPERIMENT — result

**Plan:** DEV-03 (single page, "Strangschema Sanitär"). Chosen because its
entire legend is only 7 entries, all 7 already independently confirmed by
prior human review as real, isolated, graphical potable-water component
symbols — the best available "high-quality legend" candidate among the 20
dev plans.

**Selected symbols (5 of DEV-03's 7 legend entries, chosen for diversity,
all pre-confirmed real by independent human review before this experiment
started):** Wasserzähler M 3, Rückschlagventil, Absperrklappe,
Sicherheitsventil, Rückschlagklappe.

## Method (summary — see app/symbol_reid_experiment/ for full code)

1. Built a structural template for each symbol from its own legend
   `symbol_bbox`'s raw vector geometry (line/curve/rect counts, rotation-
   normalized angle histogram, aspect ratio, true interior crossings,
   curviness — reusing `app.legend_structural_benchmark.vector_features`
   unmodified). A template is refused if it does not itself pass the
   already-validated `graphical_symbol_gate` from the legend-symbol dataset
   work.
2. Scanned the ENTIRE page's own vector primitives with a two-scale
   sliding window (no candidate generator, no text-seeded regions): a
   cheap item-count prefilter, then every surviving window's own
   fingerprint must ALSO pass `graphical_symbol_gate` (excludes text, bare
   pipe runs, dimension lines, table borders by construction) before being
   scored against each template with `structural_similarity` (threshold
   0.72). Overlapping high-score windows merged by non-max suppression.
3. Nearby text and a pipe-boundary-crossing count were attached to
   surviving matches as corroborating evidence only, never used to produce
   a match.
4. Predictions frozen (`predictions.json`, `contact_sheet.png`, per-symbol
   crops) BEFORE any manual inspection of the plan for ground truth.
   Determinism verified by running the full search twice independently and
   diffing the serialized output: **IDENTICAL**, byte-for-byte (527,224
   bytes both runs).

## Manual ground-truth inspection (done AFTER freezing predictions)

All 5 selected component types appear ONLY at the plan's single house
connection ("Hausanschluss") detail cluster near the bottom of the sheet —
every floor-level fixture row (checked directly) shows only generic
appliance icons (bathtub, washing machine, sink, water heater), never
these 5 symbols. Manually comparing each legend reference icon against
that one cluster at high zoom:

| Symbol | Actual visible occurrences (manual count) | Detected | Missed | False positives |
|---|---|---|---|---|
| Wasserzähler M 3 | 2 | 283 | 0 (both locations fall within the flood of detections, but cannot be reliably distinguished from noise by score — see below) | ≥281 |
| Rückschlagventil | 2–3 | 68 | 0–1 | ≥65 |
| Absperrklappe | not confidently identified beyond its own legend row at this zoom (visually near-identical to Rückschlagklappe's diagonal-line+dot icon; a human reviewer could not confidently tell the two apart in the schema without much higher zoom than used elsewhere in this plan) | 77 | n/a | ≥76 |
| Sicherheitsventil | 1 clearly labeled ("SV 3/4\""), possibly 1 more at the boiler branch | 85 | 0 | ≥83 |
| Rückschlagklappe | not confidently identified beyond its own legend row (same ambiguity as Absperrklappe) | 106 | n/a | ≥105 |

**The frozen contact sheet itself makes the failure visually obvious
without needing exact ground truth**: for "Wasserzähler M 3", the six
highest-scoring detections (score 0.85, tied) are dimension-line callouts
reading "120", "160", "- 2\"", "17" — plain room/pipe dimension numbers
with an incidental L-shaped tick mark, not water meters. For
"Sicherheitsventil", one of the top matches (score 0.78) is a large
cross-hatch fill pattern (a wall or insulation hatch texture). Several
matches across every row are colored multi-line pipe-riser bundle
fragments unrelated to any of the 5 symbols. A few genuinely valve-like
bowtie/triangle shapes DO appear among the results (mostly in the
Rückschlagventil and Absperrklappe rows), proving the target geometry is
not entirely undetectable — but they are interspersed with, and do not
consistently outscore, the false positives: similarity score does not
separate true from false matches into two distinguishable populations: the
top of every row already contains obvious noise.

## Why: structural fingerprints are not enough at this scale

`graphical_symbol_gate` is well validated at telling "some real 2D
component icon" from "no icon at all" (that is what it was built and
checked for). Fed a real, dense CAD drawing at full-page scale, it happily
signs off on hundreds of small compact regions that ARE genuine, real 2D
shapes — dimension-callout tick marks, hatch texture, riser-diagram
junctions, other unrelated apparatus — because a coarse aggregate
fingerprint (item counts, one rotation-normalized angle histogram,
aspect ratio, a crossing count, curviness) collapses a huge amount of
genuinely different local geometry into a small number of scalar/vector
statistics. At the ~14-23pt window sizes these symbols occupy, there are
only a handful of primitives to begin with, so many structurally-unrelated
small clusters land inside the same coarse-statistic neighborhood as a
real valve/meter icon purely by chance. This is a fundamentally different,
and harder, problem than the one `graphical_symbol_gate` was built to
solve (yes/no on a single already-located candidate region), and a
different, and harder, problem than the earlier legend-structural
benchmark solved (matching a fixed list of already-detected v0.1
candidates against legend icons, never searching raw page geometry from
scratch).

## Verdict

**STOP HANDCRAFTED SYMBOL MATCHING — TRAINED DETECTOR REQUIRED**

Even for symbols explicitly drawn and named in the plan's own legend, with
an already-validated conservative geometry gate reused as a pre-filter,
full-page deterministic vector-geometry subgraph search produces 20-140x
more "occurrences" than plausibly exist, and similarity ranking does not
reliably separate the few genuine matches from the flood of false ones
(the highest-scoring match for one symbol was a dimension-line number).
This is a clean, decisive negative result, consistent with (and stronger
than) the earlier candidate-filtering structural benchmark's own
conclusion ("TRAINED DETECTOR REQUIRED", 417d0b6). No further development
on handcrafted vector-geometry subgraph matching for this task is
warranted.
