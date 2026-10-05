# Multi-Agent v1 — Performance Baseline & Batching/Concurrency Design

Written against the first real live run (Render → GasWaterAI-Engine →
Base44 AI Gateway → InvokeLLM → Engine), HTTP 200, `execution_mode: serial`,
`total_duration_ms` ≈ 825 705 ms (≈ 13 min 46 s). Every number below that is
not explicitly attributed to the live run is derived directly from the code
as it stood before this epic (`app/multi_agent_v1/pipeline.py` §
`run_multi_agent_v1`, `app/multi_agent_v1/router.py`, each agent module) —
nowhere estimated.

## 1. Per-agent baseline: subjects, call count, input size, dependencies

| Agent | Subjects come from (router.py) | Why exactly 1 model call per subject (pre-epic code) | Input size per call | Reads another AGENT's output? |
|---|---|---|---|---|
| `planstruktur_agent` | `_route_planstruktur`: one subject per page | `classify_page()` called once per page in a plain `for` loop; legend-coverage/keyword tiers only skip the call, never batch it | 1 low-DPI full-page PNG (`render_full_page_png`, 110 dpi) + short text | No — reads `context.all_pages()`/`page.text_spans` (deterministic) |
| `symbol_agent` | `_route_symbol`: one subject per inventory item not already `COMPONENT_FACT` | `identify()` called once per subject in a `for` loop | 1 crop (`render_crop_png`, default margin) + candidate-label list as text | No — reads `context.inventory` (deterministic, built in `component_evidence.py` before any agent runs) |
| `text_agent` | `_route_text`: one subject per text span association.py left `"unassigned"` | `associate()` called once per subject in a `for` loop | 1 crop (margin 1.5) + the text string itself | No — reads `page.associations` (deterministic, from `app/plan_analysis/association.py`) |
| `leitungs_agent` | `_route_leitungs`: one subject per edge with nearby dimension-label text | Text-pattern tier (regex over `nearby_text`) resolves the obvious KW/WW/Zirkulation cases with **zero** calls; only a genuinely ambiguous edge falls through to `classify_medium()`'s one model call, called once per such subject in a `for` loop. **Live baseline: 54 calls, ≈426 s** — the single largest contributor to total runtime. | 1 crop (margin 0.8) + short text | No — reads `context.facts_for_claim_type("leitung_dimension_evidence")` (deterministic PlanFacts) |
| `anschluss_agent` | `_route_anschluss`: one subject per component with no resolved graph association | `classify_connection()` called once per subject, no deterministic pre-check at all (every routed subject needs the model). **Live baseline: 11 calls, ≈227 s** | 1 crop (margin 1.0) + fixed instruction text | No — reads `context.inventory`'s `graph_node_ids`/`graph_edge_ids`/`resolution` (deterministic, from `component_evidence.py`) |
| `schlaufungs_agent` | `_route_schlaufungs`: one subject per node/path PlanFacts left genuinely `UNRESOLVED` | `is_resolved()` belt-and-suspenders check (zero calls when already resolved) protects against router bugs, but router already only routes UNRESOLVED subjects, so in practice every routed subject reaches the one model call in `classify()`. **Live baseline: 10 calls, ≈88 s** | 1 crop (margin 1.2) + fixed instruction text | No — reads `context.is_resolved(subject_id, "schlaufung_topology")` (deterministic PlanFacts) |
| `sicherungs_agent` | `_route_sicherungs`: one subject per inventory item with `safety_device_evidence=True` | `liquid_category` is read by a pure regex over `nearby_text` with **zero** calls; `position`/`zuordnung` always need the model (`assess()`, one call per subject) since no deterministic signal exists for those two fields | 1 crop (margin 1.0) + known-device-type text | No — reads `context.inventory` + its own deterministic regex (`extract_explicit_category`) |
| `stagnations_agent` | `_route_stagnations`: one subject per `cycle_or_dead_end` inventory item | **Never calls a model at all** — `assess()` only reads already-computed `RULE-001`/`RULE-002` results | none | No — reads `context.facts_for_subject` (deterministic rule-check results) |
| `rueckfluss_agent` | `_route_rueckfluss`: one subject per `safety_device_evidence=True` inventory item (mirrors `sicherungs_agent`) | **Never calls a model at all** — `assess()` only reads `sicherungseinrichtung.liquid_category`/`category_source`/`device_type` | none | **Yes** — reads `sicherung_values[subject_id]`, populated from `sicherungs_agent`'s own `AgentObservation.value` for the SAME subject_id, in the SAME run (see pipeline.py) |
| `zirkulations_hydraulik_agent` | `_route_zirkulation`: one subject per `medium == "Zirkulation"` inventory item | Text-pattern tier (Heizband/Begleitheizung vs. circulation-pump/valve regex) resolves most cases with zero calls; only genuinely ambiguous subjects reach `classify()`'s one model call | 1 crop (margin 0.9) + fixed instruction text (temperature never read) | No — reads `context.nearby_text` + `context.inventory` (deterministic) |
| `probenahme_agent` | `_route_probenahme`: one subject per page/text-span matching a sampling-point regex | `required` is always a fixed, honest `NOT_ASSESSABLE` (no model call needed for that field); `present`/`position`/`zuordnung` always need the model (`assess()`, one call per subject) | 1 crop (margin 1.0) + fixed instruction text | No — reads `context.nearby_text`/`page.text_spans` (deterministic) |
| `nachweis_agent` | `route_nachweis()`: one subject per unresolved gap collected from every OTHER agent's observations | **Never calls a model at all** — pure regex classification of an existing error/reason string (`classify_gap_reason`) | none | **Yes, by construction** — its subjects don't exist until every other agent (Stage A + Stage B) has already produced its observations; this is not a data read of one specific agent, it is the gap-collection step itself |

## 2. Dependency graph (derived from the code above, not assumed)

Every agent except two reads **only** already-computed deterministic data
(`context.inventory` / `context.plan_facts` / `page.text_spans` /
`page.associations`), all built once in `app/main.py`'s
`deterministic_preprocessing` block **before** `run_multi_agent_v1()` is
even called. None of the remaining ten reads another agent's
`AgentObservation`. The two real, code-verified exceptions:

- **`rueckfluss_agent` → `sicherungs_agent`**: reads `sicherungs_agent`'s own
  observation `.value` for the identical `subject_id`, populated while that
  agent runs in the same request.
- **`nachweis_agent` → everyone**: its subjects are the gaps collected from
  every other agent's `observations` entries; it cannot start before all of
  them are done.

This gives exactly three stages (implemented in `pipeline.py`):

- **Stage A** — `planstruktur_agent`, `symbol_agent`, `text_agent`,
  `leitungs_agent`, `anschluss_agent`, `schlaufungs_agent`,
  `sicherungs_agent`, `stagnations_agent`, `zirkulations_hydraulik_agent`,
  `probenahme_agent`: 10 agents, provably independent of each other,
  dispatched onto a bounded `ThreadPoolExecutor`.
- **Stage B** — `rueckfluss_agent` (needs Stage A's `sicherungs_agent` output).
  No model call either way.
- **Stage C** — `nachweis_agent` (needs every Stage A/B observation). No
  model call either way.

This is flatter than the illustrative 4-stage sketch in the original task
description (which assumed `anschluss_agent`/`schlaufungs_agent` depend on
`symbol_agent`/`leitungs_agent` output) — the actual code shows no such
data flow; both read only deterministic `context.inventory`/`plan_facts`,
so they belong in Stage A alongside everything else.

## 3. Batching design

See `app/multi_agent_v1/base_agent.py::run_batched()` for the shared
mechanism every one of the 9 model-dependent agents' own `*_batch()`
method delegates to (fachliche parsing stays in each agent's own file —
only the transport is shared): per-subject cache lookup first, then
subjects are chunked by `batch_size` (subject count) AND
`MAX_IMAGES_PER_BATCH = 8` (Base44 Gateway's documented image limit),
whichever binds first; one request per chunk, `tool_schema` wrapped to
return `{"results": [{"subject_id", ...}, ...]}`; results are mapped back
by `subject_id`, never position; a subject missing from a response becomes
`UNRESOLVED`/`UNKNOWN` (never fabricated); unknown extra ids are dropped
and counted in diagnostics; a whole chunk's provider-level failure marks
only that chunk's subjects unavailable (no retry loop, no partial
fabrication).

Per-agent default batch size (`GASWATERAI_BATCH_SIZE_<AGENT_ID>` to
override), chosen conservatively per section 4 of the task and the actual
per-subject payload weight:

| Agent | Default batch size | Rationale |
|---|---|---|
| `planstruktur_agent` | 4 | whole-page overview images are heavier than a small crop |
| `symbol_agent` | 8 | small crop, single-label answer |
| `text_agent` | 8 | small crop, single-label answer |
| `leitungs_agent` | 8 | small crop, single-label answer (the 54-call agent) |
| `anschluss_agent` | 6 | moderate crop complexity |
| `schlaufungs_agent` | 6 | moderate crop complexity |
| `sicherungs_agent` | 5 | richer per-subject answer (position + zuordnung) |
| `zirkulations_hydraulik_agent` | 6 | moderate crop complexity |
| `probenahme_agent` | 8 | small crop, three simple fields |

All are ≤ the hard 8-image cap, so the configured value is always the
effective chunk size for today's one-image-per-subject agents.

## 4. Model calls: live baseline vs. theoretical new total (same plan)

| Agent | Baseline calls (live run) | New batch size | Theoretical new calls (⌈calls ÷ batch_size⌉) |
|---|---|---|---|
| `symbol_agent` | 11 | 8 | 2 |
| `leitungs_agent` | 54 | 8 | 7 |
| `anschluss_agent` | 11 | 6 | 2 |
| `schlaufungs_agent` | 10 | 6 | 2 |

No number here is a target or a promise — it is `⌈baseline ÷ configured
batch_size⌉`, the direct arithmetic consequence of this epic's batching
change on the EXACT same live plan's subject counts, to be confirmed by
the next real live run's own timing diagnostics.

## 5. Concurrency

`app/multi_agent_v1/pipeline.py` dispatches Stage A's 10 agents onto one
shared `concurrent.futures.ThreadPoolExecutor(max_workers=GASWATERAI_AGENT_MAX_CONCURRENCY)`
(default 3) — each agent's own batched calls stay sequential within that
agent (one chunk at a time), so the executor's `max_workers` directly
bounds the number of provider calls in flight across the WHOLE Stage A at
once, never just per agent. Stage B and Stage C run synchronously after
Stage A's barrier (and after Stage A+B respectively) since they make no
model calls at all. Merge order into the final `observations` list is
fixed (`_STAGE_A_AGENT_ORDER` in `pipeline.py`), independent of which
agent's thread happens to finish first — see
`tests/multi_agent_v1/test_pipeline_concurrency_and_stages.py`.
