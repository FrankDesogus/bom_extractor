from __future__ import annotations

from .models import RawRowRecord
from .utils import looks_like_code, looks_like_header, looks_like_item, looks_like_quantity, normalize_space


UOM_TOKENS = {"NR", "PZ", "KG", "M", "MM", "CM", "SET", "MT"}


def weak_map_columns(row: RawRowRecord) -> RawRowRecord:
    cols = [normalize_space(c) for c in row.extracted_columns if normalize_space(c)]
    row.extracted_columns = cols

    if not row.raw_text:
        row.raw_text = " | ".join(cols)

    if not cols:
        return row

    if looks_like_header(row.raw_text):
        row.warnings.append("header_like_content")
        return row

    start_idx = 0
    if cols and looks_like_item(cols[0]):
        row.item = row.item or cols[0]
        start_idx = 1

    working = cols[start_idx:]
    if not working:
        return row

    code_idx = None
    for i, token in enumerate(working):
        if looks_like_code(token):
            code_idx = i
            break

    if code_idx is not None:
        type_tokens = working[:code_idx]
        if type_tokens:
            row.type_raw = row.type_raw or " ".join(type_tokens)
        row.code = row.code or working[code_idx]
        after_code = working[code_idx + 1 :]
    else:
        row.type_raw = row.type_raw or working[0]
        after_code = working[1:]

    if after_code:
        rev_candidate = after_code[0]
        if len(rev_candidate) <= 6:
            row.revision = row.revision or rev_candidate
            after_code = after_code[1:]

    qty_idx = None
    uom_idx = None
    for i in range(len(after_code) - 1, -1, -1):
        token = after_code[i]
        if qty_idx is None and looks_like_quantity(token):
            qty_idx = i
            continue
        if qty_idx is not None and uom_idx is None and token.upper() in UOM_TOKENS:
            uom_idx = i
            break

    if qty_idx is not None:
        row.quantity_raw = row.quantity_raw or after_code[qty_idx]
    if uom_idx is not None:
        row.uom = row.uom or after_code[uom_idx]

    desc_parts: list[str] = []
    notes_parts: list[str] = []
    for i, token in enumerate(after_code):
        if uom_idx is not None and i == uom_idx:
            continue
        if qty_idx is not None and i == qty_idx:
            continue
        if qty_idx is not None and i > qty_idx:
            notes_parts.append(token)
            continue
        desc_parts.append(token)

    if desc_parts:
        row.description = row.description or " | ".join(desc_parts)
    if notes_parts:
        row.notes = row.notes or " | ".join(notes_parts)

    if row.item is None and row.extracted_columns and not looks_like_item(row.extracted_columns[0]):
        row.warnings.append("continuation_candidate")

    return row


def stitch_multiline_rows(rows: list[RawRowRecord]) -> list[RawRowRecord]:
    if not rows:
        return rows

    stitched: list[RawRowRecord] = []
    for row in rows:
        if stitched and "continuation_candidate" in row.warnings and "header_row" not in row.warnings:
            prev = stitched[-1]
            prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
            prev.extracted_columns.extend(row.extracted_columns)
            prev.warnings.append("row_stitched_from_multiline")
            continue
        stitched.append(row)
    return stitched
