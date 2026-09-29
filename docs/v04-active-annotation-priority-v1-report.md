# Active Annotation v1 — Positive Component Data Collection

## Goal

387 human annotations exist so far, heavily imbalanced (296 of 387 are
`NOT_A_COMPONENT`). This round does not train anything; it re-ranks the
remaining unlabeled, in-scope candidates so the hosted annotation tool
serves the reviewer the most promising candidates first, to collect more
POSITIVE drinking-water component examples with less wasted effort.

Every existing annotation is preserved unchanged. No unlabeled candidate
was ever assigned a class by this work — verified below.

## 1. Current positive-class counts (387 labeled)

| Class | Count |
|---|---|
| NOT_A_COMPONENT | 296 |
| küche | 26 |
| OTHER_RELEVANT_SYMBOL | 22 |
| wc up | 20 |
| dusche | 10 |
| up verteiler unter wt | 9 |
| AMBIGUOUS | 3 |
| secomat | 1 |
| pex-verteiler, BA | 0 |

Real positive components (excluding NOT_A_COMPONENT and AMBIGUOUS): **88**
out of 387 (22.7%). `secomat` (1), `up verteiler unter wt` (9) and `dusche`
(10) are the thinnest classes and the ones most worth targeting.

## 2. How the remaining candidates were re-ranked

`scripts/apply_annotation_priority_v1.py` computes, for every UNLABELED,
in-scope (`scope_status=IN_SCOPE`, `dedup_status=CANONICAL`) candidate, an
`annotation_priority` integer score and a `priority_reasons` list — purely
from data already in `manifest.json` (the v0.3 legend-matcher's own
suggestion label + score, the v0.1 graph relation, and bbox geometry).
**No PDF was reprocessed and no new candidate was generated.**

**How the 387 existing labels were used** (and not used): only to (a)
derive `positive_class_counts` above, (b) identify suggestion-generation
patterns with an extremely high `NOT_A_COMPONENT` rate (≥90%, n≥3 — same
method as the earlier dataset-cleanup pass), to deprioritize other
unlabeled candidates sharing that same generated label, and (c) find the
graph position of already-confirmed real components, to mildly boost other
unlabeled candidates nearby on the same plan+page. **At no point is a label
value copied onto, or used to decide, any individual unlabeled candidate's
class** — enforced by an assertion in the script itself and independently
verified after publishing (see section 6).

Score rules (weights fixed before computing, not tuned per candidate):

| Rule | Weight | Evidence |
|---|---|---|
| `graph_endpoint_or_branch` | +3 | v0.1 graph relation is `at_endpoint`/`at_branch` |
| `potable_water_terminology` | +3 | suggestion label names kalt-/warmwasser, Zirkulation, Sanitär, etc. |
| `apparatus_component_terminology` | +3 | suggestion label names a valve/tap/meter/apparatus term |
| `usable_legend_evidence` | +2 | v0.3 found a matching legend entry at all |
| `component_like_geometry` | +2 | bbox aspect ratio ≤3 (not an elongated pipe run) |
| `proximity_to_proven_potable_graph` | +2 | shares a graph node/edge, or sits close, to an already-confirmed real component on the same plan+page |
| `high_confidence_candidate_generation_evidence` | +1 | v0.3 match score ≥0.5 |
| `dimension_or_lu_label` | −3 | label is a bare LU/dimension code |
| `pure_pipe_segment_geometry` | −3 | very elongated bbox with no apparatus/potable term |
| `text_only_cluster_weak_evidence` | −2 | no suggestion at all + weakest graph tie (`near_not_connected`) |
| `insulation_or_material_description` | −3 | label names insulation/pipe material (Dämmung, Armaflex, Cr/PE-S/FL codes) |
| `titleblock_or_boilerplate_text` | −3 | label matches plan-metadata boilerplate (Plan-Name, Maßstab, Rev., "bauseits", …) |
| `known_low_value_pattern` | −4 | label is one of the 12 patterns below (≥90% NOT_A_COMPONENT, n≥3, derived fresh from the current 387) |

Patterns identified as low-value (deprioritize, never excluded outright):
`cns28 (27 lu)` (98%, n=49), `warmwasserleitung von haus 7 cr 35` (92%,
n=12), and 10 further patterns at 100% NOT_A_COMPONENT (`a`, `b`,
`(bauseits)`, `wasserversorgung`, `pe-s 110`, `messfühler niveau`, plan
scale/revision boilerplate, etc.) — full list with exact rates in
`data/dev_plans_v04/annotation_dataset/annotation_priority_v1.json`.

**Result**: 130 unlabeled candidates land in `priority_tier="high"`
(top-ranked, deterministic tie-break by score then candidate_id), 298 in
`"normal"`, 22 stay `"excluded"` (unchanged wastewater/duplicate exclusion).

Top-ranked examples (score, label): `zirkulationsregelventil` (13),
`thermostatischer mischer` (11), `wasserzähler` (×4, 10 each),
`batterieventil` (×3, 10 each), `entleerhahn` (10), `up-ventil` (10),
`feinfilter mit umgehung` (10). Bottom of the deprioritized pool:
`(bauseits)` (−5 to −8), `- 1:50 massstab: rev. a:` (−5), bare LU codes (−4).

## 3. Hosted tool changes

Three explicit views added, replacing the single flat queue:

- **Hohe Priorität** (default view) — the 130 `priority_tier="high"`
  candidates, sorted by `annotation_priority` descending.
- **Alle verbleibenden** — all 428 unlabeled in-scope candidates (high +
  normal), same sort.
- **Bereits annotiert** — every already-labeled/skipped candidate, always
  reachable, unaffected by scope/priority (unchanged guarantee from the
  earlier dataset-cleanup round).

The current candidate's `annotation_priority` and `priority_reasons` are
now shown in the tool itself (not just the backend), so the reviewer can
see why a candidate was surfaced. Wastewater/duplicate candidates remain
excluded from both unlabeled views exactly as before — they only ever
appear via "Bereits annotiert" if a human already labeled one.

Published to `https://claude.ai/artifact/NVn2g1hNY48sn291JzUAaW` (version
6).

## 4. Analysis of the 22 existing `OTHER_RELEVANT_SYMBOL` annotations

Using each one's `annotator_note` (never guessed, never inferred beyond
what the human wrote):

| Real component (from human notes) | Count |
|---|---|
| Verteilbatterie (distribution manifold) | 3 |
| Zirkulationsventile (circulation valves) | 3 |
| Badewannenmischer (bathtub mixer tap, incl. 1 typo "Badewannenmischrf") | 3 |
| Waschmaschine (washing-machine connection) | 2 |
| Gartenventil Frostsicher (frost-proof outdoor tap) | 2 |
| Sicherheitsventil (safety valve) | 1 |
| Duschenmischer (shower mixer tap) | 1 |
| Waschtrog (laundry sink) | 1 |
| "Beschriftung Verteilung" (a label/annotation, not a component itself) | 1 |
| No note, no suggestion label (unclear from available evidence) | 4 |
| No note, but a suggestion label with no keyword match | 1 |

**Recommended new explicit classes** (well-supported, n≥2, real drinking-
water-relevant fixtures distinct from the current 10-class taxonomy):
- `verteilbatterie` (3 instances)
- `zirkulationsventile` (3 instances)
- `badewannenmischer` (3 instances, including a typo variant to normalize)
- `waschmaschine` (2 instances)
- `gartenventil` (2 instances)

**Not yet recommended** (real, but only 1 instance each — insufficient
evidence to commit a new taxonomy class yet): Sicherheitsventil,
Duschenmischer, Waschtrog. Worth revisiting once this round adds more
examples.

**No relabeling was performed.** These 22 candidates keep their existing
`OTHER_RELEVANT_SYMBOL` class and notes untouched; this is a
recommendation for the taxonomy going forward, not a retroactive change —
consistent with the earlier taxonomy-cleanup task's own rule of never
silently changing existing labels.

## 5. Stopping criterion for the next annotation round

Target: **~130 additional annotations** (the full "Hohe Priorität" queue;
within the requested 100–150 range), not all 428 remaining candidates.

Stop the round when ANY of the following is true:
1. The "Hohe Priorität" queue is empty (all 130 reviewed) — the primary,
   expected stopping point.
2. **20 consecutive** high-priority candidates in a row come back
   `NOT_A_COMPONENT` — a signal the remaining high-priority pool has
   degraded faster than expected; re-evaluate the ranking rather than
   grinding through the rest.
3. At least **40 new real positive labels** (non-`NOT_A_COMPONENT`,
   non-`AMBIGUOUS`) have been collected AND every currently-thin class
   (`secomat`, `up verteiler unter wt`, `dusche`) has gained at least 3
   new examples — a yield-based early-stop if the queue turns out more
   productive than expected.

After stopping, re-run `scripts/apply_annotation_priority_v1.py` against
the updated annotation set before starting a further round — it is a pure,
deterministic function of the current data and produces a fresh queue.

## 6. Verification (performed after publishing)

- **All 387 existing annotations unchanged**: fresh dump taken after
  publishing, diffed byte-for-byte field-by-field against a dump taken
  before this task started — 0 missing, 0 new, 0 content diffs, all still
  at document `version: 1`.
- **No unlabeled candidate auto-labeled**: same diff proves this (0 new
  documents in the `annotations` collection); additionally,
  `apply_annotation_priority_v1.py` asserts at runtime that every
  candidate it processes as "unlabeled" still has `class: null` in
  `manifest.json`.
- **Wastewater/duplicate exclusion intact**: 0 of the 22 `OUT_OF_SCOPE_WASTEWATER`
  or non-canonical-duplicate candidates were given a `priority_tier` of
  `high`/`normal` — confirmed by direct query against the published
  `candidates.json`.
- **Priority ordering deterministic**: re-running
  `apply_annotation_priority_v1.py` against the same inputs a second time
  produced 0 differences in `annotation_priority`/`priority_reasons`/
  `priority_tier` across all 837 candidates.
- **Persistent storage still works**: a real write, read-back comparison,
  and delete round-trip against the live `annotations` collection
  succeeded (test document, removed immediately after).
- **Backup/export still works**: the same `ArtifactData` `list`+`out_dir`
  mechanism used for backup/audit throughout this project was used to take
  both the before- and after-change dumps above; both succeeded.

No detector was trained. Base44 was not modified. No new source plan was
processed.
