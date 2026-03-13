from __future__ import annotations

from .models import RawRowRecord
from .utils import looks_like_footer, looks_like_header, looks_like_item


def validate_row(row: RawRowRecord) -> RawRowRecord:
    text = row.raw_text or ""
    if not text.strip():
        row.warnings.append("empty_raw_text")
    if looks_like_header(text):
        row.warnings.append("header_row")
    if looks_like_footer(text):
        row.warnings.append("footer_row")
    if row.item and not looks_like_item(row.item):
        row.warnings.append("suspicious_item")
    if not row.extracted_columns:
        row.warnings.append("no_columns_extracted")
    elif len(row.extracted_columns) < 3:
        row.warnings.append("column_count_below_expected")
    return row
