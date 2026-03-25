from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from ..models import RawRowRecord
from ..provenance import (
    mark_row_merge,
    record_stage_diff,
    set_final_operational_confidence,
    snapshot_tracked_fields,
)
from ..utils import looks_like_code, looks_like_item, looks_like_quantity, normalize_space

MAX_VERTICAL_GAP = 18.0
MAX_STITCHED_FRAGMENTS = 3
MAX_PARENT_LOOKBACK = 6
LANE_TOLERANCE = 26.0
ASSIGNMENT_TOLERANCE = 34.0
LANE_AMBIGUITY_MARGIN = 16.0

SOFT_WARNINGS = {
    "continuation_candidate",
    "lane_ambiguity",
    "field_assignment_uncertain",
}

STRUCTURAL_ROW_STATES = {
    "clean_anchor_row",
    "anchor_row_with_expandable_continuation",
    "continuation_fragment_row",
    "ambiguous_row",
    "table_header_row",
    "non_bom_structural_row",
}

STRONG_WARNING_SIGNALS = {
    "anchor_lane_conflict",
    "orphan_continuation_fragment",
    "parent_row_uncertain",
    "continuation_attachment_uncertain",
    "header_row",
    "probable_header_leakage",
    "ambiguous_row_boundary",
    "parser_disagreement",
}

UOM_TOKENS = {"NR", "PZ", "KG", "KGM", "M", "MM", "CM", "SET", "MT", "EA"}


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


@dataclass(frozen=True)
class TokenSpan:
    index: int
    text: str
    x0: float
    x1: float
    x_center: float


@dataclass(frozen=True)
class AnchorIntegrity:
    code_plausibility: float
    revision_plausibility: float
    code_revision_separation_confidence: float
    uom_quantity_coherence: float
    score: float

    @property
    def code_is_plausible(self) -> bool:
        return self.code_plausibility >= 0.55

    @property
    def revision_is_plausible(self) -> bool:
        return self.revision_plausibility >= 0.6

    @property
    def separation_is_strong(self) -> bool:
        return self.code_revision_separation_confidence >= 0.58


@dataclass(frozen=True)
class LeftToRightAnchorReconstruction:
    item: str | None
    type_raw: str | None
    code: str | None
    revision: str | None
    description: str | None
    uom: str | None
    quantity_raw: str | None
    code_token_indices: tuple[int, ...]
    revision_token_index: int | None
    description_token_indices: tuple[int, ...]
    later_token_indices: tuple[int, ...]
    code_token_count: int
    description_start_index: int | None
    code_revision_boundary_confidence: float
    field_order_locked: bool
    field_order_violation_suspected: bool
    description_anchor_leakage_suspected: bool


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


def _token_spans(row: RawRowRecord) -> list[TokenSpan]:
    word_boxes = row.metadata.get("word_boxes") if isinstance(row.metadata, dict) else None
    if isinstance(word_boxes, list) and word_boxes:
        spans: list[TokenSpan] = []
        for idx, box in enumerate(word_boxes):
            text = normalize_space(str(box.get("text", "")))
            if not text:
                continue
            x0 = float(box.get("x0", 0.0))
            x1 = float(box.get("x1", x0))
            spans.append(TokenSpan(index=len(spans), text=text, x0=x0, x1=x1, x_center=(x0 + x1) / 2.0))
        return spans
    centers = _token_centers(row)
    if not centers:
        return []
    inferred_width = 12.0
    return [
        TokenSpan(index=idx, text=text, x0=center - inferred_width / 2.0, x1=center + inferred_width / 2.0, x_center=center)
        for idx, (text, center) in enumerate(centers)
    ]


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


def _find_code_start_index(tokens: list[TokenSpan]) -> int | None:
    for token in tokens:
        if _code_shape_score(token.text) >= 0.6 or looks_like_code(token.text):
            return token.index
    return None


def _find_quantity_index(tokens: list[TokenSpan], code_start_idx: int | None) -> int | None:
    lower_bound = 0 if code_start_idx is None else code_start_idx + 1
    for token in reversed(tokens):
        if token.index < lower_bound:
            break
        if not looks_like_quantity(token.text):
            continue
        has_left_uom = token.index > lower_bound and tokens[token.index - 1].text.upper() in UOM_TOKENS
        if has_left_uom:
            return token.index
    return None


def _find_uom_index(tokens: list[TokenSpan], code_start_idx: int | None, quantity_idx: int | None) -> int | None:
    lower_bound = 0 if code_start_idx is None else code_start_idx + 1
    upper_bound = len(tokens) if quantity_idx is None else quantity_idx
    for token in reversed(tokens[lower_bound:upper_bound]):
        if token.text.upper() in UOM_TOKENS:
            return token.index
    return None


def _find_revision_candidates(tokens: list[TokenSpan], code_start_idx: int | None, stop_idx: int) -> list[TokenSpan]:
    if code_start_idx is None:
        return []
    candidates: list[TokenSpan] = []
    for token in tokens[code_start_idx + 1 : stop_idx]:
        text = normalize_space(token.text)
        if not text or text.upper() in UOM_TOKENS:
            continue
        if _plausible_revision(text):
            candidates.append(token)
    return candidates


def _structural_anchor_samples(row: RawRowRecord) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {role: [] for role in ("item", "type", "code", "revision", "description", "uom", "quantity")}
    tokens = _token_spans(row)
    if not tokens:
        return samples

    code_start_idx = _find_code_start_index(tokens)
    quantity_idx = _find_quantity_index(tokens, code_start_idx)
    uom_idx = _find_uom_index(tokens, code_start_idx, quantity_idx)
    stop_idx = min(idx for idx in (uom_idx, quantity_idx, len(tokens)) if idx is not None)
    revision_candidates = _find_revision_candidates(tokens, code_start_idx, stop_idx)

    first = tokens[0]
    if looks_like_item(first.text):
        samples["item"].append(first.x_center)

    if code_start_idx is not None:
        code_token = tokens[code_start_idx]
        samples["code"].append(code_token.x_center)
        for token in tokens[:code_start_idx]:
            if token.text.isalpha() and token.text.upper() not in UOM_TOKENS:
                samples["type"].append(token.x_center)
        if revision_candidates:
            samples["revision"].append(revision_candidates[-1].x_center)
        for token in tokens[code_start_idx + 1 : stop_idx]:
            if token not in revision_candidates and len(token.text) >= 4 and not looks_like_quantity(token.text):
                samples["description"].append(token.x_center)

    if uom_idx is not None:
        samples["uom"].append(tokens[uom_idx].x_center)
    if quantity_idx is not None:
        samples["quantity"].append(tokens[quantity_idx].x_center)
    return samples


def infer_page_lane_model(rows: list[RawRowRecord]) -> PageLaneModel:
    role_samples: dict[str, list[float]] = {role: [] for role in ("item", "type", "code", "revision", "description", "uom", "quantity", "notes", "trade_name", "company_name")}
    for row in rows:
        structural_samples = _structural_anchor_samples(row)
        for role in ("item", "type", "code", "revision", "description", "uom", "quantity"):
            role_samples[role].extend(structural_samples.get(role, []))

        for token, center in _token_centers(row):
            low = token.lower()
            if looks_like_item(token) and center not in structural_samples["item"]:
                role_samples["item"].append(center)
            if _code_shape_score(token) >= 0.6 and center not in structural_samples["code"]:
                role_samples["code"].append(center)
            if token.upper() in UOM_TOKENS and center not in structural_samples["uom"]:
                role_samples["uom"].append(center)
            if any(k in low for k in ("note", "remark", "spec", "finish", "hardware")):
                role_samples["notes"].append(center)
            if any(k in low for k in ("trade", "brand", "marca")):
                role_samples["trade_name"].append(center)
            if any(k in low for k in ("supplier", "company", "fornitore", "srl", "spa", "inc", "gmbh", "ltd", "llc")):
                role_samples["company_name"].append(center)
            if token.isalpha() and len(token) <= 8 and token.upper() not in UOM_TOKENS and center not in structural_samples["type"]:
                role_samples["type"].append(center)
            if len(token) >= 4 and not looks_like_quantity(token) and center not in structural_samples["description"]:
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


def _lane_center(lane_model: PageLaneModel, role: str) -> float | None:
    lane = lane_model.lanes.get(role)
    return lane.x_center if lane is not None else None


def _distance_to_lane(token: TokenSpan, lane_model: PageLaneModel, role: str) -> float:
    center = _lane_center(lane_model, role)
    if center is None:
        return 1e9
    return abs(token.x_center - center)


def _code_revision_cutoff(lane_model: PageLaneModel, code_token: TokenSpan) -> float:
    code_center = _lane_center(lane_model, "code")
    revision_center = _lane_center(lane_model, "revision")
    if code_center is not None and revision_center is not None and revision_center > code_center + 6:
        return (code_center + revision_center) / 2.0
    return max(code_token.x1 + 18.0, code_token.x_center + 20.0)


def _later_revision_exists(tokens: list[TokenSpan], current_idx: int, stop_idx: int, lane_model: PageLaneModel) -> bool:
    for token in tokens[current_idx + 1 : stop_idx]:
        if not _plausible_revision(token.text):
            continue
        code_dist = _distance_to_lane(token, lane_model, "code")
        rev_dist = _distance_to_lane(token, lane_model, "revision")
        if rev_dist <= code_dist + 8 or code_dist == 1e9:
            return True
    return False


def _description_lane_start(lane_model: PageLaneModel, revision_token: TokenSpan | None, code_end: TokenSpan) -> float:
    description_center = _lane_center(lane_model, "description")
    revision_center = _lane_center(lane_model, "revision")
    candidate_floor = revision_token.x1 + 2.0 if revision_token is not None else code_end.x1 + 8.0
    if description_center is None:
        return candidate_floor
    if revision_center is not None and description_center <= revision_center:
        return candidate_floor
    return max(candidate_floor, min(description_center - 18.0, candidate_floor + 18.0))


def _stronger_description_evidence(token: TokenSpan, lane_model: PageLaneModel) -> bool:
    if not normalize_space(token.text):
        return False
    if token.text.upper() in UOM_TOKENS:
        return False
    if _code_shape_score(token.text) >= 0.6:
        return False
    description_dist = _distance_to_lane(token, lane_model, "description")
    code_dist = _distance_to_lane(token, lane_model, "code")
    revision_dist = _distance_to_lane(token, lane_model, "revision")
    if description_dist == 1e9:
        return any(ch.isalpha() for ch in token.text)
    return bool(
        token.x_center >= (_lane_center(lane_model, "revision") or token.x_center)
        and description_dist + 8.0 < min(code_dist, revision_dist)
    )


def _select_code_tokens(
    tokens: list[TokenSpan],
    code_start_idx: int,
    stop_idx: int,
    lane_model: PageLaneModel,
) -> tuple[list[TokenSpan], float, bool]:
    code_start = tokens[code_start_idx]
    cutoff = _code_revision_cutoff(lane_model, code_start)
    code_tokens = [code_start]
    boundary_uncertain = False

    if code_start_idx + 1 >= stop_idx:
        return code_tokens, cutoff, boundary_uncertain

    candidate = tokens[code_start_idx + 1]
    gap = max(0.0, candidate.x0 - code_start.x1)
    text = normalize_space(candidate.text)
    if not text or text.upper() in UOM_TOKENS or gap > 22.0:
        return code_tokens, cutoff, boundary_uncertain
    if not _plausible_revision(text):
        return code_tokens, cutoff, boundary_uncertain
    if _stronger_description_evidence(candidate, lane_model):
        return code_tokens, cutoff, boundary_uncertain

    code_dist = _distance_to_lane(candidate, lane_model, "code")
    rev_dist = _distance_to_lane(candidate, lane_model, "revision")
    before_cutoff = candidate.x_center <= cutoff + 6.0
    trailing_revision_exists = _later_revision_exists(tokens, candidate.index, stop_idx, lane_model)
    closer_to_code = code_dist <= rev_dist - 6.0
    near_code_lane = code_dist <= max(18.0, rev_dist)

    if before_cutoff and trailing_revision_exists and (closer_to_code or near_code_lane):
        code_tokens.append(candidate)
    elif before_cutoff and near_code_lane and _lane_center(lane_model, "revision") is None:
        code_tokens.append(candidate)
        boundary_uncertain = True

    return code_tokens, cutoff, boundary_uncertain


def _select_revision_token(
    tokens: list[TokenSpan],
    code_end_idx: int,
    stop_idx: int,
    lane_model: PageLaneModel,
) -> tuple[TokenSpan | None, bool]:
    revision_token: TokenSpan | None = None
    uncertain = False
    for token in tokens[code_end_idx + 1 : stop_idx]:
        if not _plausible_revision(token.text):
            if _stronger_description_evidence(token, lane_model):
                break
            continue
        if _stronger_description_evidence(token, lane_model):
            break
        code_dist = _distance_to_lane(token, lane_model, "code")
        rev_dist = _distance_to_lane(token, lane_model, "revision")
        if rev_dist <= code_dist + 6.0 or token.x_center > (_lane_center(lane_model, "revision") or token.x_center - 1):
            revision_token = token
            if rev_dist > code_dist + 2.0:
                uncertain = True
            break
        uncertain = True
        break
    return revision_token, uncertain


def _split_description_and_later_tokens(
    tokens: list[TokenSpan],
    start_idx: int,
    stop_idx: int,
    lane_model: PageLaneModel,
) -> tuple[list[TokenSpan], list[TokenSpan], bool]:
    description_tokens: list[TokenSpan] = []
    later_tokens: list[TokenSpan] = []
    locked_description = False
    leakage_suspected = False
    for token in tokens[start_idx:stop_idx]:
        text = normalize_space(token.text)
        if not text:
            continue
        if text.upper() in UOM_TOKENS:
            break
        if not locked_description:
            locked_description = True
        if _code_shape_score(text) >= 0.6:
            leakage_suspected = True
            later_tokens.append(token)
            continue
        if _plausible_revision(text):
            code_dist = _distance_to_lane(token, lane_model, "code")
            rev_dist = _distance_to_lane(token, lane_model, "revision")
            desc_dist = _distance_to_lane(token, lane_model, "description")
            if min(code_dist, rev_dist) + 6.0 < desc_dist:
                leakage_suspected = True
                later_tokens.append(token)
                continue
        description_tokens.append(token)
    return description_tokens, later_tokens, leakage_suspected


def _compose_left_to_right_reconstruction(row: RawRowRecord, lane_model: PageLaneModel) -> LeftToRightAnchorReconstruction | None:
    tokens = _token_spans(row)
    if not tokens:
        return None

    item = normalize_space(tokens[0].text) if looks_like_item(tokens[0].text) else None
    if item and item.lower() == "null":
        item = "null"
    code_start_idx = _find_code_start_index(tokens)
    if code_start_idx is None:
        return LeftToRightAnchorReconstruction(
            item=item,
            type_raw=normalize_space(" ".join(token.text for token in tokens[1:])) or None,
            code=None,
            revision=None,
            description=None,
            uom=None,
            quantity_raw=None,
            code_token_indices=(),
            revision_token_index=None,
            description_token_indices=(),
            later_token_indices=(),
            code_token_count=0,
            description_start_index=None,
            code_revision_boundary_confidence=0.2,
            field_order_locked=False,
            field_order_violation_suspected=False,
            description_anchor_leakage_suspected=False,
        )

    uom_idx, quantity_idx = _assign_uom_quantity_pair(row, tokens, code_start_idx)
    stop_idx = min(idx for idx in (uom_idx, quantity_idx, len(tokens)) if idx is not None)
    code_tokens, cutoff, boundary_uncertain = _select_code_tokens(tokens, code_start_idx, stop_idx, lane_model)
    code_end_idx = code_tokens[-1].index
    revision_token, revision_uncertain = _select_revision_token(tokens, code_end_idx, stop_idx, lane_model)

    description_start_idx = (revision_token.index + 1) if revision_token is not None else (code_end_idx + 1)
    description_tokens, later_tokens, leakage_suspected = _split_description_and_later_tokens(
        tokens,
        description_start_idx,
        stop_idx,
        lane_model,
    )
    description_start = description_tokens[0].index if description_tokens else None

    boundary_confidence = 0.34
    if code_tokens:
        boundary_confidence += 0.16
    if len(code_tokens) == 2:
        boundary_confidence += 0.18
    if revision_token is not None:
        boundary_confidence += 0.2
        if revision_token.x_center >= _description_lane_start(lane_model, revision_token, code_tokens[-1]) - 18.0:
            boundary_confidence += 0.08
        if revision_token.x_center > code_tokens[-1].x_center + 8.0:
            boundary_confidence += 0.08
    if boundary_uncertain or revision_uncertain:
        boundary_confidence -= 0.18
    if leakage_suspected:
        boundary_confidence -= 0.1

    type_tokens = tokens[1:code_start_idx] if item else tokens[:code_start_idx]
    field_order_violation_suspected = False
    if revision_token is not None:
        desc_floor = _description_lane_start(lane_model, revision_token, code_tokens[-1])
        for token in later_tokens:
            if token.x_center < desc_floor - 10.0:
                field_order_violation_suspected = True
                break

    return LeftToRightAnchorReconstruction(
        item=item,
        type_raw=normalize_space(" ".join(token.text for token in type_tokens)) or None,
        code=normalize_space(" ".join(token.text for token in code_tokens)) or None,
        revision=normalize_space(revision_token.text) if revision_token is not None else None,
        description=normalize_space(" ".join(token.text for token in description_tokens)) or None,
        uom=normalize_space(tokens[uom_idx].text) if uom_idx is not None else None,
        quantity_raw=normalize_space(tokens[quantity_idx].text) if quantity_idx is not None else None,
        code_token_indices=tuple(token.index for token in code_tokens),
        revision_token_index=revision_token.index if revision_token is not None else None,
        description_token_indices=tuple(token.index for token in description_tokens),
        later_token_indices=tuple(token.index for token in later_tokens),
        code_token_count=len(code_tokens),
        description_start_index=description_start,
        code_revision_boundary_confidence=round(max(0.0, min(1.0, boundary_confidence)), 3),
        field_order_locked=bool(description_tokens),
        field_order_violation_suspected=field_order_violation_suspected,
        description_anchor_leakage_suspected=leakage_suspected,
    )


def _assign_uom_quantity_pair(row: RawRowRecord, tokens: list[TokenSpan], code_start_idx: int | None) -> tuple[int | None, int | None]:
    quantity_idx = _find_quantity_index(tokens, code_start_idx)
    uom_idx = _find_uom_index(tokens, code_start_idx, quantity_idx)
    row.uom = normalize_space(tokens[uom_idx].text) if uom_idx is not None else None
    row.quantity_raw = normalize_space(tokens[quantity_idx].text) if quantity_idx is not None else None
    return uom_idx, quantity_idx


def _anchor_integrity(row: RawRowRecord) -> AnchorIntegrity:
    code_text = normalize_space(row.code or "")
    code_core = code_text.split()[0] if code_text else ""
    code_plausibility = _code_shape_score(code_core)
    if code_text and len(code_text.split()) > 1 and all(_plausible_revision(part) for part in code_text.split()[1:]):
        code_plausibility = min(1.0, code_plausibility + 0.12)

    revision_plausibility = 1.0 if _plausible_revision(row.revision) else 0.0
    separation_confidence = 0.2
    reconstruction = row.metadata.get("anchor_reconstruction") if isinstance(row.metadata, dict) else None
    if isinstance(reconstruction, dict):
        boundary_confidence = reconstruction.get("code_revision_boundary_confidence")
        if isinstance(boundary_confidence, (float, int)):
            separation_confidence = float(boundary_confidence)
    if revision_plausibility == 0.0:
        separation_confidence = min(separation_confidence, 0.35)

    if row.uom and row.quantity_raw:
        uom_quantity_coherence = 1.0 if row.uom != row.quantity_raw and row.uom.upper() in UOM_TOKENS and looks_like_quantity(row.quantity_raw) else 0.35
    elif row.uom or row.quantity_raw:
        uom_quantity_coherence = 0.45
    else:
        uom_quantity_coherence = 0.7

    score = round(
        (code_plausibility * 0.4)
        + (revision_plausibility * 0.25)
        + (separation_confidence * 0.2)
        + (uom_quantity_coherence * 0.15),
        3,
    )
    return AnchorIntegrity(
        code_plausibility=round(code_plausibility, 3),
        revision_plausibility=round(revision_plausibility, 3),
        code_revision_separation_confidence=round(separation_confidence, 3),
        uom_quantity_coherence=round(uom_quantity_coherence, 3),
        score=score,
    )


def _apply_anchor_integrity_warnings(row: RawRowRecord) -> None:
    integrity = _anchor_integrity(row)
    row.metadata["anchor_integrity"] = integrity.__dict__

    for warning in (
        "code_revision_boundary_uncertain",
        "anchor_field_reconstruction_uncertain",
        "anchor_field_integrity_low",
        "uom_quantity_pair_incomplete",
        "description_anchor_leakage_suspected",
        "field_order_violation_suspected",
    ):
        if warning in row.warnings:
            row.warnings.remove(warning)

    if (row.uom and not row.quantity_raw) or (row.quantity_raw and not row.uom):
        row.warnings.append("uom_quantity_pair_incomplete")

    if integrity.code_revision_separation_confidence < 0.58:
        row.warnings.append("code_revision_boundary_uncertain")
    if integrity.score < 0.78:
        row.warnings.append("anchor_field_reconstruction_uncertain")
    if integrity.score < 0.55:
        row.warnings.append("anchor_field_integrity_low")

    reconstruction = row.metadata.get("anchor_reconstruction") if isinstance(row.metadata, dict) else None
    if isinstance(reconstruction, dict):
        if looks_like_item(row.item) and integrity.code_is_plausible and reconstruction.get("description_anchor_leakage_suspected") is True:
            row.warnings.append("description_anchor_leakage_suspected")
        if looks_like_item(row.item) and integrity.code_is_plausible and reconstruction.get("field_order_violation_suspected") is True:
            row.warnings.append("field_order_violation_suspected")


def _reconstruct_anchor_fields(row: RawRowRecord, lane_model: PageLaneModel) -> None:
    before = snapshot_tracked_fields(row)
    reconstruction = _compose_left_to_right_reconstruction(row, lane_model)
    if reconstruction is None:
        return
    if reconstruction.code is None:
        _apply_anchor_integrity_warnings(row)
        return

    row.item = reconstruction.item
    row.type_raw = reconstruction.type_raw
    row.code = reconstruction.code
    row.revision = reconstruction.revision
    row.description = reconstruction.description
    row.uom = reconstruction.uom
    row.quantity_raw = reconstruction.quantity_raw

    row.metadata["anchor_reconstruction"] = {
        "code_tokens": row.code.split() if row.code else [],
        "code_token_indices": list(reconstruction.code_token_indices),
        "code_token_count": reconstruction.code_token_count,
        "revision_token": reconstruction.revision,
        "revision_token_index": reconstruction.revision_token_index,
        "description_token_indices": list(reconstruction.description_token_indices),
        "later_token_indices": list(reconstruction.later_token_indices),
        "description_start_index": reconstruction.description_start_index,
        "field_order_locked": reconstruction.field_order_locked,
        "field_order_violation_suspected": reconstruction.field_order_violation_suspected,
        "description_anchor_leakage_suspected": reconstruction.description_anchor_leakage_suspected,
        "code_revision_boundary_confidence": reconstruction.code_revision_boundary_confidence,
    }
    token_indices = sorted(
        set(
            list(reconstruction.code_token_indices)
            + list(reconstruction.description_token_indices)
            + list(reconstruction.later_token_indices)
            + ([reconstruction.revision_token_index] if reconstruction.revision_token_index is not None else [])
        )
    )
    record_stage_diff(
        row,
        "anchor_reconstruction",
        before,
        source_token_indices=token_indices,
        lock_state_relevant=reconstruction.field_order_locked,
    )
    _apply_anchor_integrity_warnings(row)


def _assign_fields_by_lanes(row: RawRowRecord, lane_model: PageLaneModel) -> None:
    before = snapshot_tracked_fields(row)
    spans = _token_spans(row)
    if not spans:
        return
    locked = row.metadata.get("anchor_reconstruction") if isinstance(row.metadata, dict) else {}
    locked_indices = set()
    if isinstance(locked, dict):
        for key in ("code_token_indices", "description_token_indices", "later_token_indices"):
            values = locked.get(key)
            if isinstance(values, list):
                locked_indices.update(v for v in values if isinstance(v, int))
        for key in ("revision_token_index",):
            value = locked.get(key)
            if isinstance(value, int):
                locked_indices.add(value)
    tokens = [(span, span.x_center) for span in spans if span.index not in locked_indices]
    uncertain = False
    for span, x in tokens:
        token = span.text
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
            if not row.metadata.get("anchor_reconstruction", {}).get("field_order_locked"):
                row.description = _append_field_value(row.description, token)
        if certainty < 0.35:
            uncertain = True

    if uncertain and "field_assignment_uncertain" not in row.warnings:
        row.warnings.append("field_assignment_uncertain")

    if row.uom and row.quantity_raw and row.uom == row.quantity_raw and "field_assignment_uncertain" not in row.warnings:
        row.warnings.append("field_assignment_uncertain")

    _promote_revision_from_quantity_slot(row)
    record_stage_diff(
        row,
        "lane_assignment",
        before,
        source_fragments=[span.text for span in spans],
        source_token_indices=[span.index for span in spans],
        lock_state_relevant=bool(row.metadata.get("anchor_reconstruction", {}).get("field_order_locked")),
    )


def _promote_revision_from_quantity_slot(row: RawRowRecord) -> None:
    if row.revision or not _plausible_revision(row.quantity_raw):
        return
    if row.uom and row.quantity_raw and row.uom != row.quantity_raw:
        return
    row.revision = normalize_space(row.quantity_raw)
    row.quantity_raw = None


def _plausible_revision(value: str | None) -> bool:
    token = normalize_space(value or "")
    return bool(token and len(token) <= 4 and token.isalnum() and any(ch.isdigit() for ch in token))


def _has_right_side_continuation_hint(row: RawRowRecord) -> bool:
    lane_model = row.metadata.get("page_lane_model") if isinstance(row.metadata, dict) else None
    if not isinstance(lane_model, dict) or not row.bbox_row:
        return False
    lanes = lane_model.get("lanes") if isinstance(lane_model.get("lanes"), dict) else {}
    notes_lane = lanes.get("notes")
    desc_lane = lanes.get("description")
    if not isinstance(notes_lane, dict) and not isinstance(desc_lane, dict):
        return False
    row_center = (row.bbox_row[0] + row.bbox_row[2]) / 2.0
    candidates: list[float] = []
    for lane in (notes_lane, desc_lane):
        if isinstance(lane, dict) and isinstance(lane.get("x_center"), (float, int)):
            candidates.append(float(lane["x_center"]))
    if not candidates:
        return False
    right_lane = max(candidates)
    fragments = row.metadata.get("stitched_fragments", []) if isinstance(row.metadata, dict) else []
    return row_center > right_lane + 40 and bool(fragments)


def _expandable_lane_alignment(row: RawRowRecord) -> float:
    lane_model = row.metadata.get("page_lane_model") if isinstance(row.metadata, dict) else None
    if not isinstance(lane_model, dict) or not row.bbox_row:
        return 0.0
    lanes = lane_model.get("lanes") if isinstance(lane_model.get("lanes"), dict) else {}
    center_x = (row.bbox_row[0] + row.bbox_row[2]) / 2.0
    distances: list[float] = []
    for role in ROLE_MODEL.multi_line_expandable_fields:
        lane = lanes.get(role)
        lane_x = lane.get("x_center") if isinstance(lane, dict) else None
        if isinstance(lane_x, (float, int)):
            distances.append(abs(center_x - float(lane_x)))
    if not distances:
        return 0.0
    best = min(distances)
    return round(max(0.0, 1.0 - (best / (ASSIGNMENT_TOLERANCE + 22))), 3)


def _has_unresolved_continuation_fragments(row: RawRowRecord) -> bool:
    fragments = row.metadata.get("stitched_fragments", []) if isinstance(row.metadata, dict) else []
    if isinstance(fragments, list) and fragments:
        return False
    if row.item and row.code and _plausible_revision(row.revision):
        strong_fragment_signals = {"orphan_continuation_fragment", "parent_row_uncertain", "continuation_attachment_uncertain"}
        return any(warning in row.warnings for warning in strong_fragment_signals)
    return any(
        warning in row.warnings
        for warning in (
            "continuation_candidate",
            "orphan_continuation_fragment",
            "parent_row_uncertain",
            "continuation_attachment_uncertain",
        )
    )


def _has_strong_lane_conflict(row: RawRowRecord, lane_model: PageLaneModel) -> bool:
    return "anchor_lane_conflict" in row.warnings or _lane_ambiguity_is_strong(row, lane_model)


def _looks_like_table_header_row(row: RawRowRecord) -> bool:
    atomic = row.metadata.get("atomic_line") if isinstance(row.metadata, dict) else None
    return bool(
        isinstance(atomic, dict)
        and atomic.get("is_header_like") is True
        and atomic.get("starts_with_item_anchor") is not True
    )


def _looks_like_non_bom_structural_row(row: RawRowRecord) -> bool:
    atomic = row.metadata.get("atomic_line") if isinstance(row.metadata, dict) else None
    return bool(
        "footer_row" in row.warnings
        or "probable_footer_contamination" in row.warnings
        or (isinstance(atomic, dict) and atomic.get("is_footer_like") is True)
    )


def _row_structure_classification(row: RawRowRecord) -> str:
    lane_model_dict = row.metadata.get("page_lane_model") if isinstance(row.metadata, dict) else {}
    lane_model = PageLaneModel(
        lanes={},
        lane_confidence_score=float(lane_model_dict.get("lane_confidence_score", 0.0)) if isinstance(lane_model_dict, dict) else 0.0,
        lane_overlaps=list(lane_model_dict.get("lane_overlaps", [])) if isinstance(lane_model_dict, dict) else [],
        lane_ambiguity_roles=list(lane_model_dict.get("lane_ambiguity_roles", [])) if isinstance(lane_model_dict, dict) else [],
    )
    integrity = _anchor_integrity(row)
    anchor_ok = bool(looks_like_item(row.item) and integrity.code_is_plausible and integrity.revision_is_plausible)
    has_continuation = bool(row.metadata.get("stitched_fragments")) if isinstance(row.metadata, dict) else False
    unresolved_fragments = _has_unresolved_continuation_fragments(row)
    conflicting_anchor_lane = _has_strong_lane_conflict(row, lane_model)
    right_fragment = _has_right_side_continuation_hint(row)
    anchor_duplication = "anchor_field_duplication_suspected" in row.warnings
    parser_conflict = any(w in row.warnings for w in ("boundary_disagreement", "parser_disagreement"))
    continuation_lane_aligned = _expandable_lane_alignment(row) >= 0.45

    if _looks_like_table_header_row(row):
        return "table_header_row"
    if _looks_like_non_bom_structural_row(row):
        return "non_bom_structural_row"
    if (
        anchor_ok
        and integrity.separation_is_strong
        and not has_continuation
        and not unresolved_fragments
        and not right_fragment
        and not conflicting_anchor_lane
        and "anchor_field_integrity_low" not in row.warnings
    ):
        return "clean_anchor_row"
    if anchor_ok and has_continuation and not conflicting_anchor_lane and not anchor_duplication:
        return "anchor_row_with_expandable_continuation"
    if conflicting_anchor_lane or "lane_ambiguity" in row.warnings or anchor_duplication or parser_conflict:
        return "ambiguous_row"
    if (not anchor_ok) and not _starts_with_item_anchor(row) and continuation_lane_aligned and not parser_conflict:
        return "continuation_fragment_row"
    return "ambiguous_row"


def _lane_ambiguity_is_strong(row: RawRowRecord, lane_model: PageLaneModel) -> bool:
    if len(lane_model.lane_ambiguity_roles) < 2:
        return False
    tokens = _token_centers(row)
    if not tokens:
        return False
    close_matches = 0
    for _, x in tokens:
        dists = sorted(abs(x - lane.x_center) for lane in lane_model.lanes.values())
        if len(dists) >= 2 and abs(dists[0] - dists[1]) <= LANE_AMBIGUITY_MARGIN:
            close_matches += 1
    return close_matches >= 2


def _apply_operational_confidence(row: RawRowRecord, lane_model: PageLaneModel) -> tuple[str, float]:
    base = float(row.parser_confidence)
    completeness = sum(1 for v in _row_anchor_status(row).values() if v)
    integrity = _anchor_integrity(row)
    base += min(0.18, completeness * 0.035)
    base += (lane_model.lane_confidence_score - 0.5) * 0.08
    base += (integrity.score - 0.6) * 0.18

    classification = _row_structure_classification(row)
    if classification == "clean_anchor_row":
        base += 0.16
    elif classification == "anchor_row_with_expandable_continuation":
        base -= 0.08
    elif classification in {"table_header_row", "non_bom_structural_row", "continuation_fragment_row"}:
        base -= 0.16
    else:
        base -= 0.2

    if "field_assignment_uncertain" in row.warnings:
        base -= 0.1
    if "code_revision_boundary_uncertain" in row.warnings:
        base -= 0.08
    if "anchor_field_integrity_low" in row.warnings:
        base -= 0.16
    if "anchor_lane_conflict" in row.warnings:
        base -= 0.18
    if any(w in row.warnings for w in ("parent_row_uncertain", "orphan_continuation_fragment", "probable_header_leakage")):
        base -= 0.18
    if "parser_disagreement" in row.warnings:
        base -= 0.08

    score = max(0.0, min(1.0, round(base, 3)))
    if classification == "clean_anchor_row" and score >= 0.74:
        return "high", score
    if classification == "anchor_row_with_expandable_continuation":
        return "medium", score
    if score < 0.45 or classification in {"ambiguous_row", "continuation_fragment_row", "table_header_row", "non_bom_structural_row"}:
        return "low", score
    return "medium", score


def _is_meaningful_field_uncertainty(row: RawRowRecord) -> bool:
    if "field_assignment_uncertain" not in row.warnings:
        return False
    status = _row_anchor_status(row)
    integrity = _anchor_integrity(row)
    if status["item"] and integrity.code_is_plausible and integrity.revision_is_plausible and integrity.separation_is_strong:
        return False
    return True


def _denoise_row_warnings(row: RawRowRecord, lane_model: PageLaneModel) -> int:
    suppressed = 0
    classification = _row_structure_classification(row)
    has_attached = bool(row.metadata.get("stitched_fragments")) if isinstance(row.metadata, dict) else False
    integrity = _anchor_integrity(row)

    if classification == "clean_anchor_row":
        for warning in (
            "continuation_candidate",
            "field_assignment_uncertain",
            "lane_ambiguity",
            "code_revision_boundary_uncertain",
            "anchor_field_reconstruction_uncertain",
            "anchor_field_integrity_low",
            "description_anchor_leakage_suspected",
            "field_order_violation_suspected",
        ):
            if warning in row.warnings:
                row.warnings.remove(warning)
                suppressed += 1

    if "continuation_candidate" in row.warnings and not has_attached and classification == "clean_anchor_row":
        row.warnings.remove("continuation_candidate")
        suppressed += 1

    if classification == "table_header_row":
        for warning in ("table_header_row", "probable_header_leakage"):
            if warning not in row.warnings:
                row.warnings.append(warning)
        for warning in ("continuation_candidate", "field_assignment_uncertain", "lane_ambiguity"):
            if warning in row.warnings:
                row.warnings.remove(warning)
                suppressed += 1
    elif classification == "non_bom_structural_row":
        for warning in ("non_bom_structural_row", "probable_footer_contamination"):
            if warning not in row.warnings:
                row.warnings.append(warning)
        for warning in ("continuation_candidate", "field_assignment_uncertain", "lane_ambiguity"):
            if warning in row.warnings:
                row.warnings.remove(warning)
                suppressed += 1
    elif classification == "continuation_fragment_row":
        for warning in ("continuation_fragment_row",):
            if warning not in row.warnings:
                row.warnings.append(warning)
        if not has_attached and "orphan_continuation_fragment" not in row.warnings:
            row.warnings.append("orphan_continuation_fragment")
        for warning in ("continuation_candidate", "field_assignment_uncertain"):
            if warning in row.warnings:
                row.warnings.remove(warning)
                suppressed += 1
    elif classification == "ambiguous_row":
        if "structural_state_ambiguous" not in row.warnings:
            row.warnings.append("structural_state_ambiguous")
        if "continuation_candidate" in row.warnings:
            row.warnings.remove("continuation_candidate")
            suppressed += 1
    elif classification == "anchor_row_with_expandable_continuation":
        if has_attached and "continuation_to_expandable_field" not in row.warnings:
            row.warnings.append("continuation_to_expandable_field")

    if "lane_ambiguity" in row.warnings and not _lane_ambiguity_is_strong(row, lane_model):
        row.warnings.remove("lane_ambiguity")
        suppressed += 1

    if "field_assignment_uncertain" in row.warnings and not _is_meaningful_field_uncertainty(row):
        row.warnings.remove("field_assignment_uncertain")
        suppressed += 1

    if integrity.score >= 0.78:
        for warning in (
            "code_revision_boundary_uncertain",
            "anchor_field_reconstruction_uncertain",
            "anchor_field_integrity_low",
            "description_anchor_leakage_suspected",
            "field_order_violation_suspected",
        ):
            if warning in row.warnings:
                row.warnings.remove(warning)
                suppressed += 1

    row.metadata["warning_severity"] = {
        warning: ("strong" if warning in STRONG_WARNING_SIGNALS else "soft")
        for warning in sorted(set(row.warnings))
    }
    row.metadata["row_structure_classification"] = classification
    return suppressed


def apply_page_lane_inference(rows: list[RawRowRecord]) -> tuple[list[RawRowRecord], dict[str, float | int]]:
    lane_model = infer_page_lane_model(rows)
    suppressed_warnings = 0
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
        _reconstruct_anchor_fields(row, lane_model)
        _assign_fields_by_lanes(row, lane_model)

        if len(lane_model.lane_ambiguity_roles) >= 2 and "lane_ambiguity" not in row.warnings:
            row.warnings.append("lane_ambiguity")

        if row.item and row.code:
            item_lane = lane_model.lanes.get("item")
            code_lane = lane_model.lanes.get("code")
            if item_lane and code_lane and abs(item_lane.x_center - code_lane.x_center) < 18:
                if "anchor_lane_conflict" not in row.warnings:
                    row.warnings.append("anchor_lane_conflict")

        suppressed_warnings += _denoise_row_warnings(row, lane_model)
        confidence_band, conf_score = _apply_operational_confidence(row, lane_model)
        row.parser_confidence = conf_score
        set_final_operational_confidence(row, conf_score)
        row.metadata["operational_confidence_band"] = confidence_band

    total_warnings = sum(len(set(r.warnings)) for r in rows)
    metrics: dict[str, float | int] = {
        "lane_count": len(lane_model.lanes),
        "lane_confidence_score": lane_model.lane_confidence_score,
        "field_assignment_uncertain_count": sum(1 for r in rows if "field_assignment_uncertain" in r.warnings),
        "anchor_lane_conflict_count": sum(1 for r in rows if "anchor_lane_conflict" in r.warnings),
        "code_revision_boundary_uncertain_count": sum(1 for r in rows if "code_revision_boundary_uncertain" in r.warnings),
        "anchor_field_reconstruction_uncertain_count": sum(
            1 for r in rows if "anchor_field_reconstruction_uncertain" in r.warnings
        ),
        "code_two_token_count": sum(
            1
            for r in rows
            if isinstance(r.metadata.get("anchor_reconstruction"), dict)
            and int(r.metadata["anchor_reconstruction"].get("code_token_count", 0)) == 2
        ),
        "description_anchor_leakage_count": sum(1 for r in rows if "description_anchor_leakage_suspected" in r.warnings),
        "field_order_violation_count": sum(1 for r in rows if "field_order_violation_suspected" in r.warnings),
        "high_anchor_integrity_row_count": sum(
            1
            for r in rows
            if isinstance(r.metadata.get("anchor_integrity"), dict) and float(r.metadata["anchor_integrity"].get("score", 0.0)) >= 0.78
        ),
        "low_anchor_integrity_row_count": sum(
            1
            for r in rows
            if isinstance(r.metadata.get("anchor_integrity"), dict) and float(r.metadata["anchor_integrity"].get("score", 0.0)) < 0.55
        ),
        "uom_quantity_pair_detected_count": sum(1 for r in rows if bool(r.uom and r.quantity_raw)),
        "rows_with_clean_anchor_alignment": sum(
            1
            for r in rows
            if r.item and r.code and "anchor_lane_conflict" not in r.warnings and "field_assignment_uncertain" not in r.warnings
        ),
        "clean_anchor_row_count": sum(1 for r in rows if r.metadata.get("row_structure_classification") == "clean_anchor_row"),
        "continuation_row_count": sum(
            1 for r in rows if r.metadata.get("row_structure_classification") == "anchor_row_with_expandable_continuation"
        ),
        "ambiguous_row_count": sum(1 for r in rows if r.metadata.get("row_structure_classification") == "ambiguous_row"),
        "continuation_fragment_count": sum(
            1 for r in rows if r.metadata.get("row_structure_classification") == "continuation_fragment_row"
        ),
        "table_header_row_count": sum(1 for r in rows if r.metadata.get("row_structure_classification") == "table_header_row"),
        "non_bom_structural_row_count": sum(
            1 for r in rows if r.metadata.get("row_structure_classification") == "non_bom_structural_row"
        ),
        "high_confidence_row_count": sum(1 for r in rows if r.metadata.get("operational_confidence_band") == "high"),
        "medium_confidence_row_count": sum(1 for r in rows if r.metadata.get("operational_confidence_band") == "medium"),
        "low_confidence_row_count": sum(1 for r in rows if r.metadata.get("operational_confidence_band") == "low"),
        "warning_density": round(total_warnings / max(1, len(rows)), 3),
        "noisy_warning_suppression_count": suppressed_warnings,
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
    if row.item and (looks_like_item(row.item) or normalize_space(row.item).lower() == "null"):
        return True
    if not row.extracted_columns:
        return False
    first_column = normalize_space(row.extracted_columns[0])
    return looks_like_item(first_column) or first_column.lower() == "null"


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
    before = snapshot_tracked_fields(prev)
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
    record_stage_diff(
        prev,
        "final_stitch",
        before,
        source_fragments=[row.raw_text, *row.extracted_columns],
        lock_state_relevant=locked,
    )



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
            mark_row_merge(prev, row, "final_stitch")
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
