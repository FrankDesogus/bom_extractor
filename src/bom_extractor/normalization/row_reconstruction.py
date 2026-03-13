from __future__ import annotations

from ..models import RawRowRecord
from ..utils import normalize_space


def stitch_multiline_rows(rows: list[RawRowRecord]) -> list[RawRowRecord]:
    """Merge probable continuation rows while preserving original evidence in metadata."""
    if not rows:
        return rows

    stitched: list[RawRowRecord] = []
    for row in rows:
        continuation = "continuation_candidate" in row.warnings and "header_row" not in row.warnings
        if stitched and continuation:
            prev = stitched[-1]
            prev.metadata.setdefault("stitched_fragments", []).append(
                {
                    "raw_text": row.raw_text,
                    "columns": list(row.extracted_columns),
                    "row_index_on_page": row.row_index_on_page,
                }
            )
            prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
            prev.extracted_columns.extend(row.extracted_columns)
            if "row_stitched" not in prev.warnings:
                prev.warnings.append("row_stitched")
            continue
        stitched.append(row)
    return stitched
