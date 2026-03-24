from __future__ import annotations

from .models import RawRowRecord
from .normalization.row_reconstruction import stitch_multiline_rows
from .provenance import record_stage_diff, snapshot_tracked_fields
from .utils import looks_like_code, looks_like_header, looks_like_item, looks_like_quantity, normalize_space

UOM_TOKENS = {"NR", "PZ", "KG", "M", "MM", "CM", "SET", "MT", "EA"}


def _lane_hint(row: RawRowRecord, role: str) -> float | None:
    lane_model = row.metadata.get("page_lane_model") if isinstance(row.metadata, dict) else None
    if not isinstance(lane_model, dict):
        return None
    lanes = lane_model.get("lanes")
    lane = lanes.get(role) if isinstance(lanes, dict) else None
    if not isinstance(lane, dict):
        return None
    value = lane.get("x_center")
    return float(value) if isinstance(value, (float, int)) else None


def _token_centers(row: RawRowRecord) -> list[tuple[str, float]]:
    word_boxes = row.metadata.get("word_boxes") if isinstance(row.metadata, dict) else None
    if isinstance(word_boxes, list) and word_boxes:
        out: list[tuple[str, float]] = []
        for box in word_boxes:
            text = normalize_space(str(box.get("text", "")))
            if text:
                x0 = float(box.get("x0", 0.0))
                x1 = float(box.get("x1", x0))
                out.append((text, (x0 + x1) / 2.0))
        return out
    return []


def _nearest_token(tokens: list[tuple[str, float]], target_x: float, predicate) -> str | None:
    candidates = [(abs(x - target_x), text) for text, x in tokens if predicate(text)]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def weak_map_columns(row: RawRowRecord) -> RawRowRecord:
    before = snapshot_tracked_fields(row)
    cols = [normalize_space(c) for c in row.extracted_columns if normalize_space(c)]
    row.extracted_columns = cols
    row.metadata.setdefault("raw_fragments", list(cols))

    if not row.raw_text:
        row.raw_text = " | ".join(cols)

    if not cols:
        return row

    token_centers = _token_centers(row)
    code_x = _lane_hint(row, "code")
    rev_x = _lane_hint(row, "revision")
    uom_x = _lane_hint(row, "uom")
    qty_x = _lane_hint(row, "quantity")

    if token_centers and code_x is not None and row.code is None:
        lane_code = _nearest_token(token_centers, code_x, looks_like_code)
        if lane_code:
            row.code = lane_code
    if token_centers and rev_x is not None and row.revision is None:
        lane_rev = _nearest_token(token_centers, rev_x, lambda t: t.isalnum() and 0 < len(t) <= 6)
        if lane_rev and not looks_like_item(lane_rev):
            row.revision = lane_rev
    if token_centers and uom_x is not None and row.uom is None:
        lane_uom = _nearest_token(token_centers, uom_x, lambda t: t.upper() in UOM_TOKENS)
        if lane_uom:
            row.uom = lane_uom
    if token_centers and qty_x is not None and row.quantity_raw is None:
        lane_qty = _nearest_token(token_centers, qty_x, looks_like_quantity)
        if lane_qty:
            row.quantity_raw = lane_qty

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

    code_idx = next((i for i, token in enumerate(working) if looks_like_code(token)), None)

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
        if 0 < len(rev_candidate) <= 6:
            row.revision = row.revision or rev_candidate
            after_code = after_code[1:]

    qty_idx = None
    uom_idx = None
    for i in range(len(after_code) - 1, -1, -1):
        token = after_code[i]
        if qty_idx is None and looks_like_quantity(token):
            qty_idx = i
        elif qty_idx is not None and uom_idx is None and token.upper() in UOM_TOKENS:
            uom_idx = i
            break

    if qty_idx is not None:
        row.quantity_raw = row.quantity_raw or after_code[qty_idx]
    if uom_idx is not None:
        row.uom = row.uom or after_code[uom_idx]

    desc_parts: list[str] = []
    notes_parts: list[str] = []
    for i, token in enumerate(after_code):
        if i in {idx for idx in [uom_idx, qty_idx] if idx is not None}:
            continue
        if qty_idx is not None and i > qty_idx:
            notes_parts.append(token)
        else:
            desc_parts.append(token)

    if desc_parts:
        row.description = row.description or " | ".join(desc_parts)
    if notes_parts:
        row.notes = row.notes or " | ".join(notes_parts)

    if row.item is None and row.extracted_columns and not looks_like_item(row.extracted_columns[0]):
        row.warnings.append("continuation_candidate")

    record_stage_diff(
        row,
        "weak_map_columns",
        before,
        source_fragments=list(row.extracted_columns),
    )
    return row
