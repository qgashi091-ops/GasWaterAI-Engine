"""Orchestrates the full plan-analysis pipeline:

    PDF bytes -> vector extraction -> pipe graph -> symbol/port detection
    -> text extraction (native or OCR) -> text-to-graph association
    -> structured JSON (schema.DocumentAnalysis)

This is the "own Analyse-Service" step in the GasWaterAI pipeline. It does
not classify symbols against the SVGW library, apply guideline rules, or
run any AI review -- that all happens downstream in Base44, which has the
symbol library, the ~150 reference cases and the guideline rule set this
service has no business duplicating.
"""
from __future__ import annotations

import pymupdf

from . import schema
from .association import associate_text_spans
from .bridging import bridge_graph
from .graph import build_pipe_graph
from .label_hints import guess_label_hint
from .symbols import detect_symbols
from .text import (
    NATIVE_TEXT_MIN_SPANS,
    RawTextSpan,
    compute_ocr_dpi,
    extract_native_text_spans,
    native_text_is_garbled,
    ocr_page,
)
from .vectors import classify_page_vectors


def _finalize_text_spans(
    page_number: int, native: list[RawTextSpan], ocr: list[RawTextSpan]
) -> tuple[list[schema.TextSpan], str]:
    if len(native) >= NATIVE_TEXT_MIN_SPANS:
        chosen, source = native, "native"
    elif ocr and native:
        chosen, source = native + ocr, "mixed"
    elif ocr:
        chosen, source = ocr, "ocr"
    elif native:
        chosen, source = native, "native"
    else:
        chosen, source = [], "none"

    spans: list[schema.TextSpan] = []
    for i, raw in enumerate(chosen):
        span_source = "ocr" if raw in ocr and raw not in native else "native"
        spans.append(
            schema.TextSpan(
                id=f"p{page_number}_t{i}",
                text=raw.text,
                bbox=raw.bbox,
                source=span_source,
                confidence=raw.confidence,
                font_size=raw.font_size,
                label_hint=guess_label_hint(raw.text),
            )
        )
    return spans, source


def analyze_page(page: "pymupdf.Page", page_number: int) -> schema.PageAnalysis:
    width, height = page.rect.width, page.rect.height
    warnings: list[str] = []

    # Text (native + OCR fallback) runs before vector extraction/classification
    # even though neither stage depends on the other's output -- they only
    # converge later in the text-to-graph association step. OCR's rasterized
    # page buffer is the single largest transient allocation in this
    # pipeline; running it before the (sometimes very large, for dense CAD
    # exports) vector primitive lists are built keeps that buffer's lifetime
    # from overlapping with theirs, which matters on a memory-constrained
    # deployment. Pure ordering change -- doesn't affect any computed value.
    native_spans = extract_native_text_spans(page)
    native_garbled = native_text_is_garbled(native_spans)
    ocr_spans: list[RawTextSpan] = []
    if len(native_spans) < NATIVE_TEXT_MIN_SPANS or native_garbled:
        ocr_spans = ocr_page(page, dpi=compute_ocr_dpi(width, height))
        if native_garbled:
            warnings.append(
                "Native text layer decodes to garbled characters (the source PDF's embedded "
                "font has no usable ToUnicode map) -- discarded native text and used OCR "
                "fallback instead."
            )
        else:
            warnings.append(
                "No reliable native text layer detected (likely vectorized/CAD-exported "
                "text or a scanned page) -- fell back to OCR."
            )

    usable_native = [] if native_garbled else native_spans
    text_spans, text_source = _finalize_text_spans(page_number, usable_native, ocr_spans)

    # classify_page_vectors() builds independent LineSegment/VectorPrimitive
    # dataclasses and never keeps a reference back into PyMuPDF's raw drawing
    # dicts, so once it returns, only the primitive *count* is still needed
    # (for the warnings/stats below) -- the raw list itself (which, for a
    # dense CAD export, holds many nested pymupdf.Point/Rect objects per
    # entry) is dropped immediately rather than held for the rest of the page.
    drawings = page.get_drawings()
    n_raw_drawings = len(drawings)
    page_vectors = classify_page_vectors(drawings, width, height)
    del drawings

    built_graph = build_pipe_graph(page_vectors.line_segments, width, height)
    symbols, dangling = detect_symbols(page_vectors.symbol_primitives, built_graph, width, height)

    built_graph, bridge_events = bridge_graph(built_graph, symbols, dangling, width, height)
    n_collinear_bridges = sum(1 for ev in bridge_events if ev.reason == "collinear_gap")
    n_symbol_bridges = sum(1 for ev in bridge_events if ev.reason == "symbol_junction")
    n_corner_bridges = sum(1 for ev in bridge_events if ev.reason == "corner_convergence")
    if bridge_events:
        degree_by_id = {n.id: n.degree for n in built_graph.nodes}
        dangling = [nid for nid in dangling if degree_by_id.get(nid, 1) == 1]
        warnings.append(
            f"Bridged {n_collinear_bridges} junction gap(s) via collinear continuation, "
            f"{n_symbol_bridges} via shared symbol evidence, and {n_corner_bridges} via corner "
            "convergence; every other dangling end was left unresolved rather than guessed -- see "
            "graph.edges[].is_bridge."
        )

    text_items = [(t.id, ((t.bbox[0] + t.bbox[2]) / 2.0, (t.bbox[1] + t.bbox[3]) / 2.0)) for t in text_spans]

    associations = associate_text_spans(text_items, symbols, built_graph, width, height)

    label_by_symbol: dict[str, tuple[str, str, float]] = {}
    text_by_id = {t.id: t.text for t in text_spans}
    for assoc in associations:
        if assoc.target_type != "symbol" or assoc.target_id is None:
            continue
        current = label_by_symbol.get(assoc.target_id)
        if current is None or (assoc.distance is not None and assoc.distance < current[2]):
            label_by_symbol[assoc.target_id] = (assoc.text_id, text_by_id[assoc.text_id], assoc.distance or 0.0)

    symbol_models = []
    for sym in symbols:
        label = label_by_symbol.get(sym.id)
        symbol_models.append(
            schema.SymbolCandidate(
                id=sym.id,
                bbox=sym.bbox,
                centroid=sym.centroid,
                primitive_count=sym.primitive_count,
                has_curves=sym.has_curves,
                area=sym.area,
                label=label[1] if label else None,
                label_text_id=label[0] if label else None,
                label_distance=label[2] if label else None,
                port_node_ids=list(sym.port_node_ids),
            )
        )

    node_models = [
        schema.GraphNode(id=n.id, point=n.point, degree=n.degree, is_endpoint=n.degree == 1)
        for n in built_graph.nodes
    ]
    edge_models = [
        schema.GraphEdge(
            id=e.id,
            source=e.source,
            target=e.target,
            length=e.length,
            color_rgb=e.color_rgb,
            stroke_width=e.stroke_width,
            polyline=e.polyline,
            is_bridge=e.is_bridge,
            bridge_reason=e.bridge_reason,
        )
        for e in built_graph.edges
    ]
    association_models = [
        schema.TextAssociation(
            text_id=a.text_id, target_type=a.target_type, target_id=a.target_id, distance=a.distance
        )
        for a in associations
    ]

    if n_raw_drawings > 50000:
        warnings.append(
            f"Very high vector primitive count ({n_raw_drawings}); symbol clustering is "
            "heuristic and should be spot-checked."
        )

    if page_vectors.background_line_segments:
        n_bg = len(page_vectors.background_line_segments)
        n_total = n_bg + len(page_vectors.line_segments)
        warnings.append(
            f"Excluded {n_bg} of {n_total} candidate line segments ({100 * n_bg / n_total:.0f}%) from the "
            "pipe graph: one color group dominates this page's total drawn line length with almost no "
            "other color present, which is far more consistent with a background/reference layer (walls, "
            "gridlines, dimension lines) than a multi-media pipe network. Not deleted -- see "
            "stats.background_line_segments."
        )

    stats = {
        "raw_drawings": n_raw_drawings,
        "line_segments": len(page_vectors.line_segments),
        "background_line_segments": len(page_vectors.background_line_segments),
        "glyph_primitives": len(page_vectors.glyph_primitives),
        "symbol_primitives": len(page_vectors.symbol_primitives),
        "graph_nodes": len(node_models),
        "graph_edges": len(edge_models),
        "symbols": len(symbol_models),
        "text_spans": len(text_spans),
        "dangling_pipe_ends": len(dangling),
        "bridged_collinear_gaps": n_collinear_bridges,
        "bridged_symbol_junctions": n_symbol_bridges,
        "bridged_corner_convergence": n_corner_bridges,
    }

    return schema.PageAnalysis(
        page_number=page_number,
        width=width,
        height=height,
        rotation=page.rotation,
        text_source=text_source,
        text_spans=text_spans,
        graph=schema.PipeGraph(nodes=node_models, edges=edge_models),
        symbols=symbol_models,
        associations=association_models,
        dangling_pipe_ends=dangling,
        warnings=warnings,
        stats=stats,
    )


def analyze_pdf_bytes(data: bytes, filename: str) -> schema.DocumentAnalysis:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        pages = [analyze_page(doc[i], page_number=i + 1) for i in range(len(doc))]
    finally:
        doc.close()

    warnings: list[str] = []
    ocr_pages = [p.page_number for p in pages if p.text_source in ("ocr", "mixed")]
    if ocr_pages:
        warnings.append(f"Pages requiring OCR fallback: {ocr_pages}")

    return schema.DocumentAnalysis(filename=filename, page_count=len(pages), pages=pages, warnings=warnings)


def analyze_pdf_file(path: str) -> schema.DocumentAnalysis:
    with open(path, "rb") as f:
        data = f.read()
    return analyze_pdf_bytes(data, filename=path.split("/")[-1])
