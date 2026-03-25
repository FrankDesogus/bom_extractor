from __future__ import annotations

from dataclasses import dataclass
import re
from statistics import median

from ..models import ParserPageResult, RawRowRecord
from ..provenance import mark_row_merge, set_boundary_adjusted_confidence
from ..utils import looks_like_footer, looks_like_header, looks_like_item, looks_like_quantity, normalize_space

ITEM_ANCHOR_PATTERN = re.compile(r"^\d{3,4}$")
MAX_CONTINUATION_GAP = 14.0
HEADER_START_PATTERNS = [
    re.compile(r"\b(item|riga)\b", re.IGNORECASE),
    re.compile(r"\b(code|codice)\b", re.IGNORECASE),
    re.compile(r"\b(qty|quantity|quantit)\b", re.IGNORECASE),
]
CONTINUATION_FRAGMENT_PATTERNS = [
    re.compile(r"^\d{1,3}[_\-/]\d{1,3}$"),
    re.compile(r"\b(srl|spa|inc|llc|ltd|gmbh|company|co\.)\b", re.IGNORECASE),
    re.compile(r"\b(trade|supplier|fornitore|manufacturer|manuf\.)\b", re.IGNORECASE),
    re.compile(r"\b(note|notes|remark|remarks|spec|specification|material|finish|hardware)\b", re.IGNORECASE),
]
CONTINUATION_COLUMN_HINTS = ["description", "note", "trade", "supplier", "company", "manufacturer"]


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
    is_header_like: bool = False
    is_footer_like: bool = False
    continuation_candidate: bool = False


def _is_item_anchor_token(token: str | None) -> bool:
    normalized = normalize_space(token or "")
    if not normalized:
        return False
    if normalized.lower() == "null":
        return True
    compact_digits = re.sub(r"[^\d]", "", normalized)
    return bool(ITEM_ANCHOR_PATTERN.match(compact_digits))


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
        if not _is_item_anchor_token(line.tokens[0]):
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
    if not _is_item_anchor_token(line.tokens[0]):
        return False
    if item_range is None:
        return True
    x = _first_token_x(line)
    if x is None:
        return False
    return item_range[0] <= x <= item_range[1]


def _is_table_header_anchor(line: AtomicLine) -> bool:
    if not line.text:
        return False
    hits = sum(1 for pattern in HEADER_START_PATTERNS if pattern.search(line.text))
    return hits >= 2


def _row_incomplete(row: RawRowRecord) -> bool:
    return not (row.item and row.code and row.quantity_raw)


def _x_aligned_for_continuation(prev: AtomicLine, current: AtomicLine, item_range: tuple[float, float] | None) -> bool:
    if item_range is not None and current.x_min <= item_range[1]:
        return False
    overlap = max(0.0, min(prev.x_max, current.x_max) - max(prev.x_min, current.x_min))
    prev_width = max(1.0, prev.x_max - prev.x_min)
    return overlap / prev_width > 0.35 or abs(current.x_min - prev.x_min) <= 18.0


def _column_overlap_ratio(current: AtomicLine, item_range: tuple[float, float] | None) -> float:
    if item_range is None:
        return 0.5
    non_item_start = item_range[1]
    width = max(1.0, current.x_max - current.x_min)
    non_item_overlap = max(0.0, current.x_max - max(non_item_start, current.x_min))
    return non_item_overlap / width


def _lexical_continuation_score(line: AtomicLine) -> float:
    text = normalize_space(line.text)
    if not text:
        return 0.0
    score = 0.0
    if any(pattern.search(text) for pattern in CONTINUATION_FRAGMENT_PATTERNS):
        score += 0.35
    if len(line.tokens) <= 4:
        score += 0.1
    if re.search(r"\d{1,3}[_\-/]\d{1,3}", text):
        score += 0.45
    lowered = text.lower()
    if any(hint in lowered for hint in CONTINUATION_COLUMN_HINTS):
        score += 0.2
    if not looks_like_quantity(text) and any(ch.isalpha() for ch in text):
        score += 0.1
    return min(1.0, score)


def _continuation_candidate_score(
    prev_line: AtomicLine | None, current: AtomicLine, item_range: tuple[float, float] | None
) -> float:
    if current.starts_with_item_anchor or current.is_header_like or current.is_footer_like:
        return 0.0
    score = 0.0
    if prev_line is not None:
        gap = max(0.0, current.y_top - prev_line.y_bottom)
        if gap <= MAX_CONTINUATION_GAP:
            score += 0.3
        elif gap <= MAX_CONTINUATION_GAP * 1.5:
            score += 0.1
    overlap_ratio = _column_overlap_ratio(current, item_range)
    if overlap_ratio >= 0.7:
        score += 0.35
    elif overlap_ratio >= 0.5:
        score += 0.2
    score += _lexical_continuation_score(current)
    if prev_line is not None and _x_aligned_for_continuation(prev_line, current, item_range):
        score += 0.2
    return min(1.0, score)


def annotate_atomic_lines(rows: list[RawRowRecord]) -> tuple[list[AtomicLine], list[str], dict[str, float | int]]:
    lines = [_to_atomic_line(r) for r in rows]
    lines.sort(key=lambda l: (l.y_top, l.row.row_index_on_page))

    item_range = infer_item_column_range(lines)
    warnings: list[str] = []
    if item_range is None:
        warnings.append("item_column_uncertain")

    anchor_count = 0
    prev_line: AtomicLine | None = None
    for line in lines:
        line.starts_with_item_anchor = _classify_item_anchor(line, item_range)
        line.is_header_like = looks_like_header(line.text) or _is_table_header_anchor(line)
        line.is_footer_like = looks_like_footer(line.text)
        continuation_score = _continuation_candidate_score(prev_line, line, item_range)
        line.continuation_candidate = continuation_score >= 0.55
        line.row.metadata["atomic_line"] = {
            "tokens": list(line.tokens),
            "y_top": line.y_top,
            "y_bottom": line.y_bottom,
            "x_min": line.x_min,
            "x_max": line.x_max,
            "line_density": dict(line.line_density),
            "starts_with_item_anchor": line.starts_with_item_anchor,
            "is_header_like": line.is_header_like,
            "is_footer_like": line.is_footer_like,
            "continuation_candidate": line.continuation_candidate,
            "continuation_candidate_score": round(continuation_score, 3),
        }
        if line.is_header_like and "probable_header_leakage" not in line.row.warnings:
            line.row.warnings.append("probable_header_leakage")
        if line.continuation_candidate and "continuation_candidate" not in line.row.warnings:
            line.row.warnings.append("continuation_candidate")
        if line.starts_with_item_anchor:
            anchor_count += 1
            if "item_anchor_detected" not in line.row.warnings:
                line.row.warnings.append("item_anchor_detected")
        prev_line = line

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


def _secondary_continuation_support(parser_results: list[ParserPageResult], current_row: RawRowRecord) -> bool:
    if not current_row.bbox_row:
        return False
    y = float(current_row.bbox_row[1])
    for result in parser_results:
        for row in result.rows:
            if row is current_row or not row.bbox_row:
                continue
            if abs(float(row.bbox_row[1]) - y) > 3.0:
                continue
            if row.item and looks_like_item(row.item):
                continue
            if any(tok in normalize_space(row.raw_text).lower() for tok in CONTINUATION_COLUMN_HINTS):
                return True
            if re.search(r"\d{1,3}[_\-/]\d{1,3}", normalize_space(row.raw_text)):
                return True
            if row.extracted_columns and not looks_like_item(row.extracted_columns[0]):
                return True
    return False


def _attachment_score(
    prev: RawRowRecord,
    prev_line: AtomicLine,
    current: RawRowRecord,
    current_line: AtomicLine,
    parser_results: list[ParserPageResult],
) -> tuple[float, bool]:
    gap = max(0.0, current_line.y_top - prev_line.y_bottom)
    score = 0.0
    if gap <= MAX_CONTINUATION_GAP:
        score += 0.25
    elif gap <= MAX_CONTINUATION_GAP * 1.5:
        score += 0.1

    if _x_aligned_for_continuation(prev_line, current_line, prev.metadata.get("item_column_range")):
        score += 0.2
    overlap_ratio = _column_overlap_ratio(current_line, prev.metadata.get("item_column_range"))
    score += 0.2 if overlap_ratio >= 0.55 else 0.05

    if _row_incomplete(prev):
        score += 0.2
    if prev.metadata.get("stitched_fragments"):
        score += 0.1

    score += _lexical_continuation_score(current_line) * 0.3

    parser_supported = _secondary_continuation_support(parser_results, current)
    if parser_supported:
        score += 0.2

    if current_line.is_header_like or current_line.is_footer_like:
        score -= 0.5

    return max(0.0, min(1.2, score)), parser_supported


def _update_row_confidence(row: RawRowRecord) -> None:
    confidence = float(row.parser_confidence)
    cols = [normalize_space(c) for c in row.extracted_columns if normalize_space(c)]
    has_item = bool(row.item) or any(looks_like_item(c) for c in cols[:2])
    has_code = bool(row.code) or any(re.search(r"[A-Z]?\d{6,}", c) for c in cols)
    has_qty_uom = bool(row.quantity_raw and row.uom)
    if has_item and has_code and has_qty_uom:
        confidence += 0.15
    if "continuation_attachment_uncertain" in row.warnings:
        confidence -= 0.2
    if "probable_header_leakage" in row.warnings or "header_row" in row.warnings:
        confidence -= 0.25
    if "boundary_disagreement" in row.warnings:
        confidence -= 0.15
    if "parser_supported_attachment" in row.warnings:
        confidence += 0.05
    row.parser_confidence = max(0.0, min(1.0, round(confidence, 3)))
    set_boundary_adjusted_confidence(row, row.parser_confidence)


def apply_row_boundary_engine(
    rows: list[RawRowRecord], parser_results: list[ParserPageResult]
) -> tuple[list[RawRowRecord], list[str], dict[str, float | int]]:
    if not rows:
        return rows, [], {"candidate_item_anchor_count": 0, "reconstructed_row_count": 0, "merge_ratio": 0.0}

    lines, warnings, metrics = annotate_atomic_lines(rows)
    secondary_anchors = _secondary_item_anchors(parser_results, rows)

    reconstructed: list[RawRowRecord] = []
    merge_events = 0

    table_started = False
    for line in lines:
        row = line.row
        if line.is_header_like and not line.starts_with_item_anchor:
            if _is_table_header_anchor(line):
                table_started = True
            reconstructed.append(row)
            continue
        if not table_started and line.starts_with_item_anchor:
            table_started = True
        if not table_started and not line.starts_with_item_anchor:
            if "probable_header_leakage" not in row.warnings:
                row.warnings.append("probable_header_leakage")
            reconstructed.append(row)
            continue
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

        if "probable_header_leakage" in prev.warnings or "header_row" in prev.warnings:
            reconstructed.append(row)
            continue

        score, parser_supported = _attachment_score(prev, prev_line, row, line, parser_results)
        if score < 0.55:
            if line.continuation_candidate and "continuation_attachment_uncertain" not in row.warnings:
                row.warnings.append("continuation_attachment_uncertain")
            reconstructed.append(row)
            continue
        if 0.55 <= score < 0.72:
            if "continuation_attachment_uncertain" not in row.warnings:
                row.warnings.append("continuation_attachment_uncertain")
            reconstructed.append(row)
            continue

        prev.metadata.setdefault("stitched_fragments", []).append({
            "raw_text": row.raw_text,
            "columns": list(row.extracted_columns),
            "row_index_on_page": row.row_index_on_page,
            "attachment_score": round(score, 3),
        })
        prev.metadata.setdefault("raw_fragments", [prev.raw_text])
        prev.metadata["raw_fragments"].append(row.raw_text)
        prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
        prev.extracted_columns.extend(row.extracted_columns)
        mark_row_merge(prev, row, "row_boundary_engine")
        if prev.bbox_row and row.bbox_row:
            prev.bbox_row = (
                min(prev.bbox_row[0], row.bbox_row[0]),
                min(prev.bbox_row[1], row.bbox_row[1]),
                max(prev.bbox_row[2], row.bbox_row[2]),
                max(prev.bbox_row[3], row.bbox_row[3]),
            )
        for tag in ("row_stitched", "multiline_row_detected", "possible_fragmentation", "continuation_attached"):
            if tag not in prev.warnings:
                prev.warnings.append(tag)
        if parser_supported and "parser_supported_attachment" not in prev.warnings:
            prev.warnings.append("parser_supported_attachment")
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
        tokens = _significant_tokens(text)
        item_tokens = [tok for tok in tokens if looks_like_item(tok)]
        qty_like_tokens = [tok for tok in tokens if looks_like_quantity(tok)]
        ref_fragments = [tok for tok in tokens if re.match(r"^\d{1,3}[_\-/]\d{1,3}$", tok)]
        if len(item_tokens) > 1 and "multi_item_row_detected" not in row.warnings:
            if len(item_tokens) > (len(qty_like_tokens) + len(ref_fragments) + 1):
                row.warnings.append("multi_item_row_detected")

    for row in reconstructed:
        _update_row_confidence(row)

    metrics["reconstructed_row_count"] = len(reconstructed)
    metrics["merge_ratio"] = merge_events / max(1, len(lines))

    if metrics["candidate_item_anchor_count"] and metrics["reconstructed_row_count"] < int(metrics["candidate_item_anchor_count"] * 0.75):
        warnings.append("row_loss_suspected")
    if metrics["merge_ratio"] > 0.3:
        warnings.append("excessive_row_merge_detected")

    return reconstructed, sorted(set(warnings)), metrics
