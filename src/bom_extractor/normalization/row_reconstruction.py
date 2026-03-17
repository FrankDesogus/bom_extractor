from __future__ import annotations

from dataclasses import dataclass

from ..models import RawRowRecord
from ..utils import looks_like_code, looks_like_item, looks_like_quantity, normalize_space

MAX_VERTICAL_GAP = 18.0
MAX_STITCHED_FRAGMENTS = 3


@dataclass(frozen=True)
class ColumnRoleModel:
    single_line_anchor_fields: tuple[str, ...] = ("item", "code", "revision", "uom", "quantity_raw")
    multi_line_expandable_fields: tuple[str, ...] = ("description", "notes", "trade_name", "company_name")


ROLE_MODEL = ColumnRoleModel()


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


def _row_anchor_status(row: RawRowRecord) -> dict[str, bool]:
    return {
        "item": bool(row.item),
        "code": bool(row.code),
        "revision": bool(row.revision),
        "uom": bool(row.uom),
        "quantity_raw": bool(row.quantity_raw),
    }


def _anchor_fields_locked(status: dict[str, bool]) -> bool:
    return status["item"] and status["code"] and status["revision"]


def _append_field_value(current: str | None, incoming: str | None) -> str | None:
    incoming_norm = normalize_space(incoming or "")
    if not incoming_norm:
        return current
    current_norm = normalize_space(current or "")
    if not current_norm:
        return incoming_norm
    if incoming_norm.lower() in current_norm.lower():
        return current
    return f"{current_norm} | {incoming_norm}"


def _guess_expandable_field_target(prev: RawRowRecord, row: RawRowRecord) -> str:
    text = normalize_space(row.raw_text).lower()
    row_center_x = None
    if row.bbox_row:
        row_center_x = (row.bbox_row[0] + row.bbox_row[2]) / 2
    prev_center_x = None
    if prev.bbox_row:
        prev_center_x = (prev.bbox_row[0] + prev.bbox_row[2]) / 2

    if any(k in text for k in ("trade", "marca", "brand")):
        return "trade_name"
    if any(k in text for k in ("company", "fornitore", "supplier", "srl", "spa", "inc", "gmbh", "ltd", "llc")):
        return "company_name"
    if any(k in text for k in ("note", "remark", "spec", "finish", "hardware")):
        return "notes"

    if row_center_x is not None and prev_center_x is not None and row_center_x > prev_center_x + 48:
        return "notes"
    return "description"


def _merge_continuation_by_roles(prev: RawRowRecord, row: RawRowRecord) -> None:
    anchor_status = _row_anchor_status(prev)
    locked = _anchor_fields_locked(anchor_status)
    if locked and "row_anchor_fields_locked" not in prev.warnings:
        prev.warnings.append("row_anchor_fields_locked")

    incoming_status = _row_anchor_status(row)
    for field in ROLE_MODEL.single_line_anchor_fields:
        incoming = getattr(row, field)
        if not incoming:
            continue
        existing = getattr(prev, field)
        if existing and normalize_space(str(existing)) != normalize_space(str(incoming)):
            if "anchor_field_duplication_suspected" not in prev.warnings:
                prev.warnings.append("anchor_field_duplication_suspected")
            if "continuation_attachment_uncertain" not in prev.warnings:
                prev.warnings.append("continuation_attachment_uncertain")
            continue
        if existing:
            continue
        if field in ("item", "code", "revision") and locked:
            if "anchor_field_duplication_suspected" not in prev.warnings:
                prev.warnings.append("anchor_field_duplication_suspected")
            continue
        if field in ("uom", "quantity_raw") and anchor_status[field]:
            if "anchor_field_duplication_suspected" not in prev.warnings:
                prev.warnings.append("anchor_field_duplication_suspected")
            continue
        setattr(prev, field, incoming)

    target = _guess_expandable_field_target(prev, row)
    if target not in ROLE_MODEL.multi_line_expandable_fields:
        target = "description"
    incoming_text = normalize_space(row.raw_text)
    setattr(prev, target, _append_field_value(getattr(prev, target), incoming_text))
    if "continuation_to_expandable_field" not in prev.warnings:
        prev.warnings.append("continuation_to_expandable_field")

    for field in ROLE_MODEL.multi_line_expandable_fields:
        incoming = getattr(row, field)
        if incoming:
            setattr(prev, field, _append_field_value(getattr(prev, field), incoming))

    if not any(incoming_status.values()) and "continuation_attachment_uncertain" not in prev.warnings:
        prev.warnings.append("continuation_attachment_uncertain")


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
            prev.metadata.setdefault("raw_fragments", [prev.raw_text])
            prev.metadata["raw_fragments"].append(row.raw_text)
            prev.raw_text = normalize_space(f"{prev.raw_text} {row.raw_text}")
            prev.extracted_columns.extend(row.extracted_columns)
            _merge_continuation_by_roles(prev, row)
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
