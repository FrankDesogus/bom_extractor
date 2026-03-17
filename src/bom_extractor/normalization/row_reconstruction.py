from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..models import RawRowRecord
from ..utils import looks_like_code, looks_like_item, looks_like_quantity, normalize_space

MAX_VERTICAL_GAP = 18.0
MAX_STITCHED_FRAGMENTS = 3
MAX_PARENT_LOOKBACK = 6
LANE_TOLERANCE = 26.0
ASSIGNMENT_TOLERANCE = 34.0

UOM_TOKENS = {"NR", "PZ", "KG", "M", "MM", "CM", "SET", "MT", "EA"}


@dataclass(frozen=True)
class ColumnRoleModel:
    single_line_anchor_fields: tuple[str, ...] = ("item", "code", "revision", "uom", "quantity_raw")
    multi_line_expandable_fields: tuple[str, ...] = ("description", "notes", "trade_name", "company_name")


ROLE_MODEL = ColumnRoleModel()


@dataclass(frozen=True)
class LaneCandidate:
    role: str
    x_center: float
    score: float
    supports: int
    x_min: float
    x_max: float


@dataclass(frozen=True)
class PageLaneModel:
    lanes: dict[str, LaneCandidate]
    lane_confidence_score: float
    lane_overlaps: list[str]
    lane_ambiguity_roles: list[str]


def _token_centers(row: RawRowRecord) -> list[tuple[str, float]]:
    word_boxes = row.metadata.get("word_boxes") if isinstance(row.metadata, dict) else None
    if isinstance(word_boxes, list) and word_boxes:
        out: list[tuple[str, float]] = []
        for box in word_boxes:
            text = normalize_space(str(box.get("text", "")))
            if not text:
                continue
            x0 = float(box.get("x0", 0.0))
            x1 = float(box.get("x1", x0))
            out.append((text, (x0 + x1) / 2.0))
        return out
    if row.bbox_row and row.extracted_columns:
        left, _, right, _ = row.bbox_row
        width = max(1.0, right - left)
        step = width / max(1, len(row.extracted_columns))
        return [
            (normalize_space(col), left + ((idx + 0.5) * step))
            for idx, col in enumerate(row.extracted_columns)
            if normalize_space(col)
        ]
    return []


def _code_shape_score(token: str) -> float:
    value = normalize_space(token)
    compact = value.replace(" ", "")
    if not compact:
        return 0.0
    length = len(compact)
    alpha = sum(1 for ch in compact if ch.isalpha())
    digits = sum(1 for ch in compact if ch.isdigit())
    if digits < 4 or length < 7:
        return 0.0
    mix_bonus = 0.3 if alpha > 0 else 0.15
    length_bonus = 0.3 if 8 <= length <= 16 else 0.12
    return min(1.0, mix_bonus + length_bonus + (digits / max(1, length)) * 0.4)


def _cluster_centers(samples: list[float], tolerance: float = LANE_TOLERANCE) -> list[tuple[float, int, float, float]]:
    if not samples:
        return []
    ordered = sorted(samples)
    clusters: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [(float(median(group)), len(group), min(group), max(group)) for group in clusters]


def infer_page_lane_model(rows: list[RawRowRecord]) -> PageLaneModel:
    role_samples: dict[str, list[float]] = {role: [] for role in ("item", "type", "code", "revision", "description", "uom", "quantity", "notes", "trade_name", "company_name")}
    for row in rows:
        for token, center in _token_centers(row):
            low = token.lower()
            if looks_like_item(token):
                role_samples["item"].append(center)
            if _code_shape_score(token) >= 0.6:
                role_samples["code"].append(center)
            if token.upper() in UOM_TOKENS:
                role_samples["uom"].append(center)
            if looks_like_quantity(token):
                role_samples["quantity"].append(center)
            if any(k in low for k in ("note", "remark", "spec", "finish", "hardware")):
                role_samples["notes"].append(center)
            if any(k in low for k in ("trade", "brand", "marca")):
                role_samples["trade_name"].append(center)
            if any(k in low for k in ("supplier", "company", "fornitore", "srl", "spa", "inc", "gmbh", "ltd", "llc")):
                role_samples["company_name"].append(center)
            if 1 <= len(token) <= 4 and token.isalnum() and any(ch.isdigit() for ch in token):
                role_samples["revision"].append(center)
            if token.isalpha() and len(token) <= 8 and token.upper() not in UOM_TOKENS:
                role_samples["type"].append(center)
            if len(token) >= 4 and not looks_like_quantity(token):
                role_samples["description"].append(center)

    lanes: dict[str, LaneCandidate] = {}
    ambiguity_roles: list[str] = []
    for role, samples in role_samples.items():
        clusters = _cluster_centers(samples)
        if not clusters:
            continue
        clusters = sorted(clusters, key=lambda c: c[1], reverse=True)
        best = clusters[0]
        support = best[1]
        score = min(1.0, (support / max(1, len(rows))) * 1.6)
        if len(clusters) > 1 and clusters[1][1] >= max(2, int(best[1] * 0.75)):
            ambiguity_roles.append(role)
            score *= 0.75
        lanes[role] = LaneCandidate(role=role, x_center=best[0], score=round(score, 3), supports=support, x_min=best[2], x_max=best[3])

    if "type" in lanes and "revision" in lanes and "code" in lanes:
        type_x = lanes["type"].x_center
        rev_x = lanes["revision"].x_center
        code = lanes["code"]
        if type_x <= code.x_center <= rev_x or rev_x <= code.x_center <= type_x:
            pass
        else:
            ambiguity_roles.append("code")
            lanes["code"] = LaneCandidate(**{**lanes["code"].__dict__, "score": round(lanes["code"].score * 0.8, 3)})

    overlaps: list[str] = []
    ordered = sorted(lanes.values(), key=lambda lane: lane.x_center)
    for left, right in zip(ordered, ordered[1:]):
        if left.x_max + 2 >= right.x_min:
            overlaps.append(f"{left.role}:{right.role}")

    confidence = sum(l.score for l in lanes.values()) / max(1, len(lanes))
    if overlaps:
        confidence *= 0.88
    if ambiguity_roles:
        confidence *= 0.85

    return PageLaneModel(
        lanes=lanes,
        lane_confidence_score=round(max(0.0, min(1.0, confidence)), 3),
        lane_overlaps=sorted(set(overlaps)),
        lane_ambiguity_roles=sorted(set(ambiguity_roles)),
    )


def _nearest_lane_role(x: float, lane_model: PageLaneModel, allowed_roles: tuple[str, ...]) -> tuple[str | None, float]:
    best_role: str | None = None
    best_dist = 1e9
    for role in allowed_roles:
        lane = lane_model.lanes.get(role)
        if lane is None:
            continue
        dist = abs(lane.x_center - x)
        if dist < best_dist:
            best_role = role
            best_dist = dist
    if best_role is None:
        return None, 0.0
    certainty = max(0.0, 1.0 - (best_dist / ASSIGNMENT_TOLERANCE))
    return best_role, round(certainty, 3)


def _set_if_missing(row: RawRowRecord, field: str, value: str) -> None:
    if not getattr(row, field):
        setattr(row, field, normalize_space(value))


def _assign_fields_by_lanes(row: RawRowRecord, lane_model: PageLaneModel) -> None:
    tokens = _token_centers(row)
    if not tokens:
        return
    uncertain = False
    for token, x in tokens:
        if looks_like_item(token):
            _set_if_missing(row, "item", token)
            continue
        if token.upper() in UOM_TOKENS:
            _set_if_missing(row, "uom", token)
            continue
        if looks_like_quantity(token):
            role, certainty = _nearest_lane_role(x, lane_model, ("quantity", "revision"))
            if role == "quantity":
                _set_if_missing(row, "quantity_raw", token)
            elif role == "revision" and not row.revision:
                _set_if_missing(row, "revision", token)
            if certainty < 0.4:
                uncertain = True
            continue
        if _code_shape_score(token) >= 0.6:
            role, certainty = _nearest_lane_role(x, lane_model, ("code", "type", "description"))
            if role == "code":
                _set_if_missing(row, "code", token)
            elif role == "type":
                _set_if_missing(row, "type_raw", token)
            else:
                row.description = _append_field_value(row.description, token)
            if certainty < 0.4:
                uncertain = True
            continue

        role, certainty = _nearest_lane_role(
            x,
            lane_model,
            ("type", "revision", "description", "notes", "trade_name", "company_name"),
        )
        if role == "type":
            _set_if_missing(row, "type_raw", token)
        elif role == "revision" and len(token) <= 6 and token.isalnum():
            _set_if_missing(row, "revision", token)
        elif role in {"notes", "trade_name", "company_name"}:
            setattr(row, role, _append_field_value(getattr(row, role), token))
        else:
            row.description = _append_field_value(row.description, token)
        if certainty < 0.35:
            uncertain = True

    if uncertain and "field_assignment_uncertain" not in row.warnings:
        row.warnings.append("field_assignment_uncertain")

    if row.uom and row.quantity_raw and row.uom == row.quantity_raw and "field_assignment_uncertain" not in row.warnings:
        row.warnings.append("field_assignment_uncertain")


def apply_page_lane_inference(rows: list[RawRowRecord]) -> tuple[list[RawRowRecord], dict[str, float | int]]:
    lane_model = infer_page_lane_model(rows)
    for row in rows:
        row.metadata["page_lane_model"] = {
            "lane_confidence_score": lane_model.lane_confidence_score,
            "lane_overlaps": lane_model.lane_overlaps,
            "lane_ambiguity_roles": lane_model.lane_ambiguity_roles,
            "lanes": {
                role: {
                    "x_center": lane.x_center,
                    "score": lane.score,
                    "supports": lane.supports,
                    "x_min": lane.x_min,
                    "x_max": lane.x_max,
                }
                for role, lane in lane_model.lanes.items()
            },
        }
        _assign_fields_by_lanes(row, lane_model)

        if lane_model.lane_ambiguity_roles and "lane_ambiguity" not in row.warnings:
            row.warnings.append("lane_ambiguity")

        if row.item and row.code:
            item_lane = lane_model.lanes.get("item")
            code_lane = lane_model.lanes.get("code")
            if item_lane and code_lane and abs(item_lane.x_center - code_lane.x_center) < 18:
                if "anchor_lane_conflict" not in row.warnings:
                    row.warnings.append("anchor_lane_conflict")

        conf = float(row.parser_confidence)
        conf += (lane_model.lane_confidence_score - 0.5) * 0.12
        if "field_assignment_uncertain" in row.warnings:
            conf -= 0.12
        if "lane_ambiguity" in row.warnings:
            conf -= 0.08
        if "anchor_lane_conflict" in row.warnings:
            conf -= 0.1
        row.parser_confidence = max(0.0, min(1.0, round(conf, 3)))

    metrics: dict[str, float | int] = {
        "lane_count": len(lane_model.lanes),
        "lane_confidence_score": lane_model.lane_confidence_score,
        "field_assignment_uncertain_count": sum(1 for r in rows if "field_assignment_uncertain" in r.warnings),
        "anchor_lane_conflict_count": sum(1 for r in rows if "anchor_lane_conflict" in r.warnings),
        "rows_with_clean_anchor_alignment": sum(
            1
            for r in rows
            if r.item and r.code and "anchor_lane_conflict" not in r.warnings and "field_assignment_uncertain" not in r.warnings
        ),
    }
    return rows, metrics


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


def _continuation_candidate(row: RawRowRecord) -> bool:
    atomic = row.metadata.get("atomic_line") if isinstance(row.metadata, dict) else None
    if isinstance(atomic, dict):
        if atomic.get("starts_with_item_anchor") is True:
            return False
        if atomic.get("is_header_like") is True or atomic.get("is_footer_like") is True:
            return False
        if atomic.get("continuation_candidate") is True:
            return True
    if "header_row" in row.warnings or "footer_row" in row.warnings:
        return False
    if _starts_with_item_anchor(row):
        return False
    return "continuation_candidate" in row.warnings


def _anchor_like_signals(row: RawRowRecord) -> int:
    cols = [normalize_space(c) for c in row.extracted_columns if normalize_space(c)]
    score = 0
    if row.item or (cols and looks_like_item(cols[0])):
        score += 2
    if row.code or any(looks_like_code(c) for c in cols):
        score += 1
    if row.revision:
        score += 1
    if row.uom:
        score += 1
    if row.quantity_raw or any(looks_like_quantity(c) for c in cols):
        score += 1
    return score


def _lane_compatibility(parent: RawRowRecord, row: RawRowRecord) -> float:
    if not parent.bbox_row or not row.bbox_row:
        return 0.5
    row_center = (row.bbox_row[0] + row.bbox_row[2]) / 2
    parent_left, _, parent_right, _ = parent.bbox_row
    if row_center < parent_left or row_center > parent_right + 80:
        return 0.0
    span = max(1.0, parent_right - parent_left)
    rel = (row_center - parent_left) / span
    return 1.0 if 0.2 <= rel <= 1.1 else 0.4


def _choose_parent_index(stitched: list[RawRowRecord], row: RawRowRecord) -> tuple[int | None, float]:
    best_idx: int | None = None
    best_score = -1.0
    lookback = 0
    for idx in range(len(stitched) - 1, -1, -1):
        parent = stitched[idx]
        lookback += 1
        if lookback > MAX_PARENT_LOOKBACK:
            break
        if parent.page_number != row.page_number:
            break
        gap = _vertical_gap(parent, row)
        if gap is not None and gap > MAX_VERTICAL_GAP * 1.6:
            continue
        lane_score = _lane_compatibility(parent, row)
        if lane_score <= 0:
            continue
        completeness = sum(1 for v in _row_anchor_status(parent).values() if v)
        score = 0.0
        if gap is not None:
            score += max(0.0, 1.0 - (gap / (MAX_VERTICAL_GAP * 1.6))) * 0.35
        score += (1.0 / max(1, len(stitched) - idx)) * 0.2
        score += lane_score * 0.2
        score += min(1.0, completeness / 5.0) * 0.25
        if score > best_score:
            best_score = score
            best_idx = idx
        if _starts_with_item_anchor(parent):
            break
    return best_idx, best_score


def _route_expandable_field(parent: RawRowRecord, row: RawRowRecord) -> str:
    text = normalize_space(row.raw_text)
    lowered = text.lower()
    if any(k in lowered for k in ("supplier", "company", "fornitore", "srl", "spa", "inc", "gmbh", "ltd", "llc")):
        return "company_name"
    if any(k in lowered for k in ("trade", "marca", "brand", "manufacturer")):
        return "trade_name"
    if any(k in lowered for k in ("note", "remark", "spec", "finish", "hardware")):
        return "notes"
    lane_model = parent.metadata.get("page_lane_model") if isinstance(parent.metadata, dict) else None
    if isinstance(lane_model, dict) and row.bbox_row:
        lanes = lane_model.get("lanes") if isinstance(lane_model.get("lanes"), dict) else {}
        center_x = (row.bbox_row[0] + row.bbox_row[2]) / 2.0
        nearest: tuple[str, float] | None = None
        for role in ROLE_MODEL.multi_line_expandable_fields:
            lane = lanes.get(role)
            if not isinstance(lane, dict):
                continue
            lane_x = lane.get("x_center")
            if not isinstance(lane_x, (float, int)):
                continue
            dist = abs(center_x - float(lane_x))
            if nearest is None or dist < nearest[1]:
                nearest = (role, dist)
        if nearest and nearest[1] <= ASSIGNMENT_TOLERANCE + 22:
            return nearest[0]
    if "_" in text and any(ch.isdigit() for ch in text):
        return "notes"
    if parent.notes and (len(text.split()) <= 5 or text[:1].islower()):
        return "notes"
    return _guess_expandable_field_target(parent, row)


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

    target = _route_expandable_field(prev, row)
    if target not in ROLE_MODEL.multi_line_expandable_fields:
        target = "description"
    incoming_text = normalize_space(row.raw_text)
    setattr(prev, target, _append_field_value(getattr(prev, target), incoming_text))
    prev.metadata[f"{target}_attachments"] = int(prev.metadata.get(f"{target}_attachments", 0)) + 1
    if "continuation_to_expandable_field" not in prev.warnings:
        prev.warnings.append("continuation_to_expandable_field")

    for field in ROLE_MODEL.multi_line_expandable_fields:
        incoming = getattr(row, field)
        if incoming:
            setattr(prev, field, _append_field_value(getattr(prev, field), incoming))

    if not any(incoming_status.values()) and "continuation_attachment_uncertain" not in prev.warnings:
        prev.warnings.append("continuation_attachment_uncertain")



def continuation_metrics(rows: list[RawRowRecord]) -> dict[str, float | int]:
    attached = 0
    isolated = 0
    uncertain = 0
    duplication = 0
    expandable_attachment_count = 0
    for row in rows:
        fragments = row.metadata.get("stitched_fragments", []) if isinstance(row.metadata, dict) else []
        if isinstance(fragments, list):
            attached += len(fragments)
        if "continuation_candidate" in row.warnings and not fragments and "parent_row_attached" not in row.warnings:
            isolated += 1
        if "continuation_attachment_uncertain" in row.warnings or "parent_row_uncertain" in row.warnings:
            uncertain += 1
        if "anchor_field_duplication_suspected" in row.warnings:
            duplication += 1
        for key in ("description_attachments", "notes_attachments", "trade_name_attachments", "company_name_attachments"):
            expandable_attachment_count += int(row.metadata.get(key, 0) if isinstance(row.metadata, dict) else 0)

    denom = max(1, attached + isolated)
    return {
        "isolated_continuation_count": isolated,
        "attached_continuation_count": attached,
        "uncertain_continuation_count": uncertain,
        "continuation_attachment_rate": round(attached / denom, 3),
        "anchor_field_duplication_events": duplication,
        "expandable_field_attachment_count": expandable_attachment_count,
    }

def stitch_multiline_rows(rows: list[RawRowRecord]) -> list[RawRowRecord]:
    """Merge likely continuation rows conservatively while preserving evidence."""
    if not rows:
        return rows

    stitched: list[RawRowRecord] = []
    merge_events = 0

    for row in rows:
        if row.item and row.code and row.revision and "continuation_candidate" in row.warnings:
            row.warnings.remove("continuation_candidate")
        continuation = _continuation_candidate(row)
        if continuation and _anchor_like_signals(row) >= 4:
            if "anchor_field_duplication_suspected" not in row.warnings:
                row.warnings.append("anchor_field_duplication_suspected")
            if "continuation_attachment_uncertain" not in row.warnings:
                row.warnings.append("continuation_attachment_uncertain")

        if stitched and continuation:
            if _starts_with_item_anchor(row):
                if "hard_merge_block_item_anchor" not in row.warnings:
                    row.warnings.append("hard_merge_block_item_anchor")
                stitched.append(row)
                continue

            parent_idx, parent_score = _choose_parent_index(stitched, row)
            if parent_idx is None or parent_score < 0.52:
                fallback_prev = stitched[-1]
                gap = _vertical_gap(fallback_prev, row)
                if gap is not None and gap > MAX_VERTICAL_GAP and "merge_blocked_vertical_gap" not in fallback_prev.warnings:
                    fallback_prev.warnings.append("merge_blocked_vertical_gap")
                elif fallback_prev.bbox_row and row.bbox_row and not _vertically_aligned(fallback_prev, row):
                    if "ambiguous_alignment" not in fallback_prev.warnings:
                        fallback_prev.warnings.append("ambiguous_alignment")
                if "orphan_continuation_fragment" not in row.warnings:
                    row.warnings.append("orphan_continuation_fragment")
                if "parent_row_uncertain" not in row.warnings:
                    row.warnings.append("parent_row_uncertain")
                stitched.append(row)
                continue
            prev = stitched[parent_idx]

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
                if "parent_row_uncertain" not in row.warnings:
                    row.warnings.append("parent_row_uncertain")
                stitched.append(row)
                continue

            aligned = _vertically_aligned(prev, row)
            lexical_reference = any(ch.isdigit() for ch in row.raw_text) and any(sep in row.raw_text for sep in ("_", "-", "/"))
            if not aligned and row.bbox_row and prev.bbox_row and not lexical_reference and _lane_compatibility(prev, row) < 0.8:
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
            if "parent_row_attached" not in prev.warnings:
                prev.warnings.append("parent_row_attached")
            if "continuation_candidate" not in row.warnings:
                row.warnings.append("continuation_candidate")
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
