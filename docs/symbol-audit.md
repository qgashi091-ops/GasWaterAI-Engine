# Symbol Library Data Audit (v0.2, prerequisite to any recognition work)

Source: the existing 58 SVGW W3 (Ausgabe 2013, Anhang 4, S. 71) symbol
reference images and metadata (`id`, `name`/`beschreibung`, `quelle`),
committed at `app/data/symbol_library/`. **Not** the W-001…W-010 holdout
plans — those are evaluation-only and were never inspected to build this
audit or the recognition thresholds below.

A critical, easy-to-miss fact that shaped the whole recognition strategy:
**the 58 reference images are themselves raster** (embedded scanned-legend
bitmaps extracted from the source PDF, not vector paths) — see
`app/plan_analysis/vectors.py`'s own classification, which has no concept
of "compare this vector path to that vector path." So even though the
*target plan* is a native vector CAD export, at least one side of every
comparison is unavoidably raster. This is why the recognition strategy
below is raster/classical-CV-based rather than vector-to-vector — it is a
forced consequence of the actual data, not a convenience choice.

## Method

For each of the 58 images: threshold to a binary ink mask, crop to the ink
bounding box, resize (aspect-preserving, padded) onto a fixed 128×128
canvas — this removes absolute scale and image-DPI differences by
construction. Computed per symbol: ink-pixel ratio, contour count, hole
count (via contour hierarchy), aspect ratio. Computed pairwise: normalized
cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`) between every
pair of canonical masks — a scale/position-invariant measure of shared
outer silhouette. Full numbers are in `app/data/symbol_library/audit.json`
(machine-readable) and `scratch_audit/` (the one-off analysis scripts, not
shipped, kept in history for reproducibility of this audit).

## 1–2. Distinctive vs. ambiguous

**43 of 58** symbols are visually distinctive enough for classical shape
matching in isolation (`audit.json: eligible_for_shape_matching`).

**7 ambiguous families** emerged from clustering every pair scoring ≥ 0.60
(a threshold chosen because every pair at or above it shares essentially
the same outer silhouette, differing only in a small internal mark neither
classical contour matching nor a human glancing at a small plan icon would
reliably catch):

| Family | Members | Why they're confusable |
|---|---|---|
| Pipe-label examples | SYM-002, 003, 004 | Each is a plain line with a different text label baked in — the *shape* is identical; only the text differs |
| Crossing lines / union fitting | SYM-005, 042 | Coincidental shape overlap (both a thin cross/line pattern); different real-world meanings |
| Plain isolation valve family | SYM-008, 013, 014 | "Absperrarmatur" (generic), "Schieber" (gate valve), "Absperrklappe" (butterfly valve) render as the **same bare bowtie glyph** in this legend (008↔014 scored 0.821, 008↔013 scored 0.793, 013↔014 scored 0.866) — the legend itself does not visually distinguish these three; real plans presumably rely on drawing convention/context/labeling this engine cannot recover from shape alone |
| Seat valve / ball valve | SYM-011, 012 | Bowtie + small internal mark, too similar (0.727) to trust without the mark being cleanly resolved |
| Arrow-only outlet family | SYM-016, 017, 018, 019 | Auslaufventil / Standauslaufventil / Wandauslaufventil / Mischbatterie are all rendered as little more than a bare arrow (ink ratio 0.014–0.022, a single contour) — essentially indistinguishable from each other **and** from generic plan clutter |
| Funnel-shaped | SYM-041, 056 | "Trichter" (funnel) and "Rohrbelüfter Bauart HD" coincidentally share a funnel silhouette (0.718) |
| Bauart-coded aerator vs. free outlet | SYM-031, 052 | Share a similar valve-with-tail silhouette (0.662) despite different real meanings |

## 3. Symbols requiring text/legend corroboration

Detected by a **rule over the metadata text**, not a hand-picked list:
flag any symbol whose `beschreibung`/`name` says a placeholder is replaced
by a letter code ("Stern wird ersetzt…") or names a specific "Bauart"
(construction/type variant identified only by a type code in practice).

**9 symbols**: SYM-001 (Wasserleitung — the generic pipe placeholder
itself), SYM-025 (Sicherungseinrichtung — the icon is generic, the actual
device type AA/AB/AC/AD/AF/BA/CA/… is a text code, not a shape),
SYM-052/053/054/055/056/057/058 (the Bauart AA…LB aerator/outlet family —
same reasoning). For all of these, a shape match at best identifies "this
is *a* member of this device family" — the specific type is a `FACT` only
when corroborated by a directly-associated text code (reusing
`label_hints.py`'s existing `safety_device_code` pattern and
`association.py`'s existing direct text-to-symbol association — no new
text infrastructure was built for this).

## 4. Duplicates / near-duplicates

The highest-scoring pairs are the strongest candidates for genuine
near-duplication rather than "just somewhat similar": SYM-016↔018 (0.874),
SYM-013↔014 (0.866), SYM-008↔014 (0.821), SYM-008↔013 (0.793). These four
pairs, plus the families above, mean roughly a fifth of the 58-symbol
library cannot be told apart by outer silhouette alone at the confidence
this engine requires for a `COMPONENT_FACT`.

## 5. Low-distinctiveness flag (independent of family clustering)

12 symbols (SYM-001, 005, 006, 016–023, 042) have either an ink ratio below
3% or a single, elongated, otherwise-featureless contour (a bare line,
arrow, or dot). These are flagged as high-false-positive-risk **against
arbitrary plan clutter** even where they don't score highly against any
*other specific* symbol in the library — a lone arrow or dot is common
noise on a real plan (dimension arrows, hatch marks, drafting dots; see
`bridging.py`'s own docstring on exactly this class of ambiguity for pipe
geometry) that this library's 58 references don't model at all.

## What this means for the recognition strategy

Of 58 catalog entries: **5 are not physical components at all** (pipe/line
notation, already fully covered by v0.1's topology and by the existing
`label_hints.py`/`association.py` text layer — SYM-001 through SYM-005),
**7 ambiguous families / 12 low-distinctiveness symbols** can at best reach
`COMPONENT_CANDIDATE`, never a bare-shape `COMPONENT_FACT`, and **9**
additionally require a directly-associated text code before any specific
type claim is a `FACT`. The remaining pool of genuinely reliable,
shape-alone-sufficient symbols is smaller than 58 — this is reported
honestly in `docs/w003-component-poc-report.md` rather than papered over
with a lowered confidence threshold.
