from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median

from ..models import ParserPageResult, RawRowRecord
from ..utils import looks_like_item, normalize_space

ITEM_ANCHOR_PATTERN = re.compile(r"^\d{3,4}$")
MAX_CONTINUATION_GAP = 14.0


@dataclass(slots=True)
class AtomicLine:
    row: RawRowRecord
    text: str
    tokens: list[str]
    word_boxes: list[dict[str, float | str]]
    y_top: float
    y_bottom: float
    x_min: float
    x_max: float
    line_density: dict[str, float]
    starts_with_item_anchor: bool = False


def _significant_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", normalize_space(text)) if t and any(ch.isalnum() for ch in t)]


def _to_atomic_line(row: RawRowRecord) -> AtomicLine:
    word_boxes = row.metadata.get("word_boxes") if isinstance(row.metadata.get("word_boxes"), list) else []
    tokens = _significant_tokens(row.raw_text)

    if word_boxes:
        x_min = min(float(w.get("x0", 0.0)) for w in word_boxes)
        x_max = max(float(w.get("x1", 0.0)) for w in word_boxes)
        y_top = min(float(w.get("y0", 0.0)) for w in word_boxes)
        y_bottom = max(float(w.get("y1", 0.0)) for w in word_boxes)
    elif row.bbox_row:
        x_min, y_top, x_max, y_bottom = row.bbox_row
    else:
        x_min = 0.0
        x_max = float(len(row.raw_text))
        y_top = float(row.row_index_on_page * 12)
        y_bottom = y_top + 10.0

    width = max(1.0, x_max - x_min)
    line_density = {
        "token_count": float(len(tokens)),
        "char_count": float(len(normalize_space(row.raw_text))),
        "tokens_per_width": float(len(tokens)) / width,
    }
    return AtomicLine(
        row=row,
        text=row.raw_text,
        tokens=tokens,
        word_boxes=word_boxes,
        y_top=float(y_top),
        y_bottom=float(y_bottom),
        x_min=float(x_min),
        x_max=float(x_max),
        line_density=line_density,
    )


def _first_token_x(line: AtomicLine) -> float | None:
    if not line.tokens:
        return None
    first = line.tokens[0]
    if line.word_boxes:
        for box in line.word_boxes:
            text = normalize_space(str(box.get("text", "")))
            if text == first:
                return float(box.get("x0", line.x_min))
    return line.x_min


def _cluster_positions(values: list[float], tolerance: float = 16.0) -> list[list[float]]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def infer_item_column_range(lines: list[AtomicLine]) -> tuple[float, float] | None:
    item_positions: list[float] = []
    for line in lines:
        if not line.tokens:
            continue
        first = re.sub(r"[^\d]", "", line.tokens[0])
        if not ITEM_ANCHOR_PATTERN.match(first):
            continue
        x = _first_token_x(line)
        if x is not None:
            item_positions.append(x)

    if len(item_positions) < 2:
        return None
    clusters = _cluster_positions(item_positions)
    strongest = max(clusters, key=len)
    center = median(strongest)
    return (float(center - 12.0), float(center + 20.0))


def _classify_item_anchor(line: AtomicLine, item_range: tuple[float, float] | None) -> bool:
    if not line.tokens:
        return False
    first_token = re.sub(r"[^\d]", "", line.tokens[0])
    if not ITEM_ANCHOR_PATTERN.match(first_token):
        return False
    if item_range is None:
        return True
    x = _first_token_x(line)
    if x is None:
        return False
    return item_range[0] <= x <= item_range[1]


def _row_incomplete(row: RawRowRecord) -> bool:
    return not (row.item and row.code and row.quantity_raw)


def _x_aligned_for_continuation(prev: AtomicLine, current: AtomicLine, item_range: tuple[float, float] | None) -> bool:
    if item_range is not None and current.x_min <= item_range[1]:
        return False
    overlap = max(0.0, min(prev.x_max, current.x_max) - max(prev.x_min, current.x_min))
    prev_width = max(1.0, prev.x_max - prev.x_min)
    return overlap / prev_width > 0.35 or abs(current.x_min - prev.x_min) <= 18.0


def annotate_atomic_lines(rows: list[RawRowRecord]) -> tuple[list[AtomicLine], list[str], dict[str, float | int]]:
    lines = [_to_atomic_line(r) for r in rows]
    lines.sort(key=lambda l: (l.y_top, l.row.row_index_on_page))

    item_range = infer_item_column_range(lines)
    warnings: list[str] = []
    if item_range is None:
        warnings.append("item_column_uncertain")

    anchor_count = 0
    for line in lines:
        line.starts_with_item_anchor = _classify_item_anchor(line, item_range)
        line.row.metadata["atomic_line"] = {
            "tokens": list(line.tokens),
            "y_top": line.y_top,
            "y_bottom": line.y_bottom,
            "x_min": line.x_min,
            "x_max": line.x_max,
            "line_density": dict(line.line_density),
            "starts_with_item_anchor": line.starts_with_item_anchor,
        }
        if line.starts_with_item_anchor:
            anchor_count += 1
            if "item_anchor_detected" not in line.row.warnings:
                line.row.warnings.append("item_anchor_detected")

    if item_range is not None:
        for line in lines:
            line.row.metadata["item_column_range"] = [item_range[0], item_range[1]]

    metrics: dict[str, float | int] = {
        "candidate_item_anchor_count": anchor_count,
    }
    return lines, warnings, metrics


def _secondary_item_anchors(parser_results: list[ParserPageResult], selected_rows: list[RawRowRecord]) -> list[float]:
    selected_parser = selected_rows[0].parser_name if selected_rows else ""
    anchors: list[float] = []
    for result in parser_results:
        if result.parser_name == selected_parser:
            continue
        for row in result.rows:
            first = row.item or (row.extracted_columns[0] if row.extracted_columns else "")
            if looks_like_item(first) and row.bbox_row:
                anchors.append(float(row.bbox_row[1]))
    return sorted(anchors)


def apply_row_boundary_engine(
    rows: list[RawRowRecord], parser_results: list[ParserPageResult]
) -> tuple[list[RawRowRecord], list[str], dict[str, float | int]]:
    if not rows:
        return rows, [], {"candidate_item_anchor_count": 0, "reconstructed_row_count": 0, "merge_ratio": 0.0}

    lines, warnings, metrics = annotate_atomic_lines(rows)
    secondary_anchors = _secondary_item_anchors(parser_results, rows)

    reconstructed: list[RawRowRecord] = []
    merge_events = 0

    for line in lines:
        row = line.row
        if not reconstructed:
            reconstructed.append(row)
            continue

        prev = reconstructed[-1]
        prev_line = next((l for l in lines if l.row is prev), None)
        if line.starts_with_item_anchor:
            if "hard_merge_block_item_anchor" not in row.warnings:
                row.warnings.append("hard_merge_block_item_anchor")
            reconstructed.append(row)
            continue

        if prev_line is None:
            reconstructed.append(row)
            continue

        gap = max(0.0, line.y_top - prev_line.y_bottom)
        can_merge = (
            not line.starts_with_item_anchor
            and gap <= MAX_CONTINUATION_GAP
            and _x_aligned_for_continuation(prev_line, line, prev.metadata.get("item_column_range"))
            and _row_incomplete(prev)
        )

        if not can_merge:
            reconstructed.append(row)
            continue

        prev.metadata.setdefault("stitched_fragments", []).append({
            "raw_text": row.raw_text,
            "columns": list(row.extracted_columns),
            "row_index_on_page": row.row_index_on_page,
        })
        prev.metadata.setdefault("raw_fragments", [prev.raw_text])
        prev.metadata["raw_fragments"].append(row.raw_text)
        prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
        prev.extracted_columns.extend(row.extracted_columns)
        if prev.bbox_row and row.bbox_row:
            prev.bbox_row = (
                min(prev.bbox_row[0], row.bbox_row[0]),
                min(prev.bbox_row[1], row.bbox_row[1]),
                max(prev.bbox_row[2], row.bbox_row[2]),
                max(prev.bbox_row[3], row.bbox_row[3]),
            )
        for tag in ("row_stitched", "multiline_row_detected", "possible_fragmentation"):
            if tag not in prev.warnings:
                prev.warnings.append(tag)
        merge_events += 1

    if secondary_anchors:
        for row in reconstructed:
            if not row.bbox_row:
                continue
            y = float(row.bbox_row[1])
            close = any(abs(y - ay) <= 5.0 for ay in secondary_anchors)
            if not close:
                if "boundary_disagreement" not in row.warnings:
                    row.warnings.append("boundary_disagreement")
        if any("boundary_disagreement" in r.warnings for r in reconstructed):
            warnings.append("boundary_disagreement")

    for row in reconstructed:
        text = normalize_space(" ".join([row.raw_text, *row.extracted_columns]))
        item_tokens = [tok for tok in _significant_tokens(text) if looks_like_item(tok)]
        if len(item_tokens) > 1 and "multi_item_row_detected" not in row.warnings:
            row.warnings.append("multi_item_row_detected")

    metrics["reconstructed_row_count"] = len(reconstructed)
    metrics["merge_ratio"] = merge_events / max(1, len(lines))

    if metrics["candidate_item_anchor_count"] and metrics["reconstructed_row_count"] < int(metrics["candidate_item_anchor_count"] * 0.75):
        warnings.append("row_loss_suspected")
    if metrics["merge_ratio"] > 0.3:
        warnings.append("excessive_row_merge_detected")

    return reconstructed, sorted(set(warnings)), metrics
