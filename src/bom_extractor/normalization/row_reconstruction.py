from __future__ import annotations

from ..models import RawRowRecord
from ..utils import looks_like_item, normalize_space


def _vertically_aligned(previous: RawRowRecord, current: RawRowRecord) -> bool:
    if not previous.bbox_row or not current.bbox_row:
        return False
    prev_x0, _, prev_x1, _ = previous.bbox_row
    curr_x0, _, curr_x1, _ = current.bbox_row
    overlap = max(0.0, min(prev_x1, curr_x1) - max(prev_x0, curr_x0))
    prev_width = max(1.0, prev_x1 - prev_x0)
    return overlap / prev_width > 0.45


def stitch_multiline_rows(rows: list[RawRowRecord]) -> list[RawRowRecord]:
    """Merge probable continuation rows while preserving original evidence in metadata."""
    if not rows:
        return rows

    stitched: list[RawRowRecord] = []
    for row in rows:
        first_col = row.extracted_columns[0] if row.extracted_columns else None
        continuation = (
            "continuation_candidate" in row.warnings
            or (not looks_like_item(first_col) and "header_row" not in row.warnings and "footer_row" not in row.warnings)
        )

        if stitched and continuation:
            prev = stitched[-1]
            aligned = _vertically_aligned(prev, row)
            if not aligned and row.bbox_row and prev.bbox_row:
                prev.warnings.append("ambiguous_alignment")
            prev.metadata.setdefault("stitched_fragments", []).append(
                {
                    "raw_text": row.raw_text,
                    "columns": list(row.extracted_columns),
                    "row_index_on_page": row.row_index_on_page,
                }
            )
            prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
            prev.extracted_columns.extend(row.extracted_columns)
            for warning in ("row_stitched", "multiline_row_detected", "possible_fragmentation"):
                if warning not in prev.warnings:
                    prev.warnings.append(warning)
            continue

        stitched.append(row)
    return stitched
