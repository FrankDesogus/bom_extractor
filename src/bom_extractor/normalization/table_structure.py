from __future__ import annotations

from statistics import median

from ..models import ParserPageResult, RawRowRecord

ROW_DROP_WARN_THRESHOLD = 0.35


def infer_column_boundaries(parser_results: list[ParserPageResult]) -> list[float]:
    """Infer dynamic x-boundaries from geometric parser metadata plus tabular hints."""
    geom_hints: list[float] = []
    col_count_hints: list[int] = []

    for result in parser_results:
        hint = result.metadata.get("column_x_hints")
        if isinstance(hint, list):
            geom_hints.extend(float(v) for v in hint)
        count_hint = result.metadata.get("column_count_hint")
        if isinstance(count_hint, int) and count_hint > 1:
            col_count_hints.append(count_hint)

    if geom_hints:
        xs = sorted(geom_hints)
        boundaries: list[float] = []
        for x in xs:
            if not boundaries or abs(boundaries[-1] - x) > 14:
                boundaries.append(x)
            else:
                boundaries[-1] = (boundaries[-1] + x) / 2
        return boundaries

    if col_count_hints:
        count = max(2, int(median(col_count_hints)))
        return [float(i) for i in range(count)]

    return []


def rebuild_columns_from_word_boxes(row: RawRowRecord, boundaries: list[float]) -> RawRowRecord:
    word_boxes = row.metadata.get("word_boxes")
    if not isinstance(word_boxes, list) or not word_boxes:
        return row
    if len(boundaries) < 2:
        return row

    cols: list[list[str]] = [[] for _ in range(len(boundaries) + 1)]
    for box in word_boxes:
        if not isinstance(box, dict):
            continue
        text = str(box.get("text", "")).strip()
        if not text:
            continue
        x0 = float(box.get("x0", 0.0))
        idx = 0
        while idx < len(boundaries) and x0 > boundaries[idx]:
            idx += 1
        cols[idx].append(text)

    collapsed = [" ".join(tokens).strip() for tokens in cols]
    row.extracted_columns = [c for c in collapsed if c]
    row.metadata["column_boundaries"] = boundaries
    row.metadata.setdefault("raw_fragments", list(row.extracted_columns))
    return row


def _build_secondary_row_anchors(parser_results: list[ParserPageResult], selected_name: str) -> list[float]:
    anchors: list[float] = []
    for result in parser_results:
        if result.parser_name == selected_name:
            continue
        for row in result.rows:
            if row.bbox_row:
                anchors.append(float(row.bbox_row[1]))
    return sorted(anchors)


def _is_boundary_supported(row: RawRowRecord, anchors: list[float]) -> bool:
    if not anchors or not row.bbox_row:
        return True
    y = float(row.bbox_row[1])
    return any(abs(y - a) <= 5.0 for a in anchors)


def apply_structure_assisted_reconstruction(
    selected_rows: list[RawRowRecord], parser_results: list[ParserPageResult]
) -> tuple[list[RawRowRecord], list[str], list[float]]:
    boundaries = infer_column_boundaries(parser_results)
    warnings: list[str] = []
    if not boundaries:
        warnings.append("irregular_column_structure")

    reconstructed: list[RawRowRecord] = []
    selected_name = parser_results[0].parser_name if parser_results else ""
    secondary_anchors = _build_secondary_row_anchors(parser_results, selected_name)

    for row in selected_rows:
        row = rebuild_columns_from_word_boxes(row, boundaries)
        row.metadata.setdefault("parser_sources", [row.parser_name])
        if not _is_boundary_supported(row, secondary_anchors):
            row.metadata["boundary_ambiguous"] = True
            if "ambiguous_row_boundary" not in row.warnings:
                row.warnings.append("ambiguous_row_boundary")
        reconstructed.append(row)

    parser_names = {r.parser_name for r in parser_results}
    if len(parser_names) > 1:
        for row in reconstructed:
            row.metadata["parser_sources"] = sorted(parser_names)

    candidate_lines = max((len(r.rows) for r in parser_results), default=len(reconstructed))
    if candidate_lines and len(reconstructed) < int(candidate_lines * (1 - ROW_DROP_WARN_THRESHOLD)):
        warnings.append("row_count_sanity_check_failed")
        warnings.append("large_row_count_drop")

    if any("ambiguous_row_boundary" in row.warnings for row in reconstructed):
        warnings.append("parser_conflict_detected")

    return reconstructed, warnings, boundaries
