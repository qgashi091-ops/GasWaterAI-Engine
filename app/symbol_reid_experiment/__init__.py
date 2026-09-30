"""ONE-PLAN SYMBOL REIDENTIFICATION EXPERIMENT.

A narrowly scoped GO/NO-GO feasibility experiment, explicitly NOT part of
the legend-symbol annotation dataset (see app/legend_symbol_dataset/) and
NOT another annotation pipeline. The single question this package answers:
if a potable-water symbol is explicitly drawn and named in one plan's own
legend, can its OWN raw vector geometry be used to find that same symbol's
occurrences elsewhere in the SAME plan's drawing area, with no candidate
generator, no learned detector, and no human-annotation tuning?

Method, in order:
1. `template.py` builds a structural TEMPLATE for a chosen legend symbol:
   its own vector primitives (reusing
   `app.legend_structural_benchmark.vector_features`, unmodified) plus its
   raw legend text as semantic identity. No text is ever used to invent a
   template where no legend symbol geometry exists.
2. `search.py` scans the ENTIRE drawing area of the same PDF page for local
   regions whose OWN vector geometry passes the same deterministic
   `app.legend_symbol_dataset.graphical_symbol_gate` (never text, never a
   bare pipe run, never a dimension/table line -- reusing that
   already-validated gate here doubles as the "must not treat X as a
   match" requirement) and whose structural fingerprint is similar enough
   to the template's (`vector_features.structural_similarity`, also
   unmodified). Only after a geometric match exists are nearby text and a
   simple pipe-boundary-crossing count attached as CORROBORATING evidence
   -- never used to produce a match on their own.
3. `scripts/run_symbol_reid_experiment.py` freezes every prediction (JSON +
   rendered crops + a legend-vs-occurrences contact sheet) BEFORE any
   manual inspection of the plan happens, then a human visually counts
   actual occurrences per symbol in the frozen PDF and the script's own
   report compares detected vs. actual.

No LLM is used anywhere in this package. No model is trained. Base44 and
the hosted annotation tool are untouched.
"""
