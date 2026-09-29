"""Legend + vector-geometry structural symbol benchmark (v0.4 phase 6d).

Evaluates a hypothesis: a plan's own legend, combined with v0.1's existing
vector geometry and pipe graph, can identify components without a trained
detector and without raster template matching (v0.2/v0.3 already showed
raster template matching alone does not generalize well).

This package is intentionally separate from `app.plan_analysis`: it only
*reads* that package's public functions (pipeline, legend_detection,
legend_entries for row/ambiguity classification) and never modifies it.

Strict human-annotation holdout: no module in this package ever imports
`ArtifactData`, reads a database dump, or otherwise touches human-authored
labels. Only `scripts/legend_benchmark_phaseC_score.py` (outside this
package) does that, and only after `scripts/legend_benchmark_phaseB_freeze.py`
has already hashed and frozen Phase A's predictions.
"""
