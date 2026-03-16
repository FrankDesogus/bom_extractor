from __future__ import annotations

from ..models import RawRowRecord
from ..utils import looks_like_code, looks_like_item, looks_like_quantity, normalize_space

MAX_VERTICAL_GAP = 18.0
MAX_STITCHED_FRAGMENTS = 3


def _vertically_aligned(previous: RawRowRecord, current: RawRowRecord) -> bool:
    if not previous.bbox_row or not current.bbox_row:
        return False
    prev_x0, _, prev_x1, _ = previous.bbox_row
    curr_x0, _, curr_x1, _ = current.bbox_row
    overlap = max(0.0, min(prev_x1, curr_x1) - max(prev_x0, curr_x0))
    prev_width = max(1.0, prev_x1 - prev_x0)
    return overlap / prev_width > 0.45


def _vertical_gap(previous: RawRowRecord, current: RawRowRecord) -> float | None:
    if not previous.bbox_row or not current.bbox_row:
        return None
    return max(0.0, current.bbox_row[1] - previous.bbox_row[3])


def _row_has_full_pattern(row: RawRowRecord) -> bool:
    cols = [normalize_space(c) for c in row.extracted_columns if normalize_space(c)]
    has_item = bool(row.item) or any(looks_like_item(c) for c in cols[:2])
    has_code = bool(row.code) or any(looks_like_code(c) for c in cols)
    has_qty_uom = (bool(row.quantity_raw) and bool(row.uom))
    if not has_qty_uom:
        has_qty = any(looks_like_quantity(c) for c in cols)
        has_uom = any(c.isalpha() and 1 <= len(c) <= 4 for c in cols)
        has_qty_uom = has_qty and has_uom
    return has_item and has_code and has_qty_uom


def _new_item_appears(row: RawRowRecord) -> bool:
    if row.item and looks_like_item(row.item):
        return True
    if not row.extracted_columns:
        return False
    return looks_like_item(row.extracted_columns[0])




def _starts_with_item_anchor(row: RawRowRecord) -> bool:
    atomic = row.metadata.get("atomic_line") if isinstance(row.metadata, dict) else None
    if isinstance(atomic, dict) and atomic.get("starts_with_item_anchor") is True:
        return True
    return _new_item_appears(row)


def stitch_multiline_rows(rows: list[RawRowRecord]) -> list[RawRowRecord]:
    """Merge likely continuation rows conservatively while preserving evidence."""
    if not rows:
        return rows

    stitched: list[RawRowRecord] = []
    merge_events = 0

    for row in rows:
        first_col = row.extracted_columns[0] if row.extracted_columns else None
        continuation = (
            "continuation_candidate" in row.warnings
            or (not looks_like_item(first_col) and "header_row" not in row.warnings and "footer_row" not in row.warnings)
        )

        if stitched and continuation:
            prev = stitched[-1]
            if _starts_with_item_anchor(row):
                if "hard_merge_block_item_anchor" not in row.warnings:
                    row.warnings.append("hard_merge_block_item_anchor")
                stitched.append(row)
                continue
            if _row_has_full_pattern(prev):
                if "merge_blocked_full_pattern" not in prev.warnings:
                    prev.warnings.append("merge_blocked_full_pattern")
                stitched.append(row)
                continue

            fragments = prev.metadata.setdefault("stitched_fragments", [])
            if len(fragments) >= MAX_STITCHED_FRAGMENTS:
                if "excessive_row_merge_detected" not in prev.warnings:
                    prev.warnings.append("excessive_row_merge_detected")
                stitched.append(row)
                continue

            gap = _vertical_gap(prev, row)
            if gap is not None and gap > MAX_VERTICAL_GAP:
                if "merge_blocked_vertical_gap" not in prev.warnings:
                    prev.warnings.append("merge_blocked_vertical_gap")
                stitched.append(row)
                continue

            aligned = _vertically_aligned(prev, row)
            if not aligned and row.bbox_row and prev.bbox_row:
                if "ambiguous_alignment" not in prev.warnings:
                    prev.warnings.append("ambiguous_alignment")
                stitched.append(row)
                continue

            if "boundary_disagreement" in row.warnings:
                stitched.append(row)
                continue

            fragments.append(
                {
                    "raw_text": row.raw_text,
                    "columns": list(row.extracted_columns),
                    "row_index_on_page": row.row_index_on_page,
                }
            )
            prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
            prev.extracted_columns.extend(row.extracted_columns)
            if prev.bbox_row and row.bbox_row:
                prev.bbox_row = (
                    min(prev.bbox_row[0], row.bbox_row[0]),
                    min(prev.bbox_row[1], row.bbox_row[1]),
                    max(prev.bbox_row[2], row.bbox_row[2]),
                    max(prev.bbox_row[3], row.bbox_row[3]),
                )
            for warning in ("row_stitched", "multiline_row_detected", "possible_fragmentation"):
                if warning not in prev.warnings:
                    prev.warnings.append(warning)
            merge_events += 1
            continue

        stitched.append(row)

    if merge_events >= max(4, int(len(rows) * 0.25)):
        for row in stitched:
            if row.metadata.get("stitched_fragments") and "excessive_row_merge_detected" not in row.warnings:
                row.warnings.append("excessive_row_merge_detected")

    return stitched
