from __future__ import annotations

from typing import Any

from .models import RawRowRecord
from .utils import normalize_space

TRACKED_FIELDS = (
    "item",
    "type_raw",
    "code",
    "revision",
    "description",
    "trade_name",
    "company_name",
    "uom",
    "quantity_raw",
    "notes",
)


def row_debug_id(row: RawRowRecord) -> str:
    return f"p{row.page_number}:r{row.row_index_on_page}"


def ensure_row_provenance(row: RawRowRecord) -> dict[str, Any]:
    metadata = row.metadata if isinstance(row.metadata, dict) else {}
    row.metadata = metadata
    diag = metadata.setdefault("diagnostic_provenance", {})
    diag.setdefault("field_provenance", {})
    diag.setdefault(
        "row_merge_provenance",
        {
            "merged_by_boundary_engine": False,
            "merged_by_final_stitch": False,
            "parent_row_ids": [row_debug_id(row)],
            "child_row_ids": [],
        },
    )
    confidence = diag.setdefault("confidence_provenance", {})
    confidence.setdefault("parser_seed_confidence", float(row.parser_confidence))
    return diag


def _field_entry(row: RawRowRecord, field: str) -> dict[str, Any]:
    diag = ensure_row_provenance(row)
    field_map = diag["field_provenance"]
    entry = field_map.setdefault(
        field,
        {
            "final_writer_stage": None,
            "touched_by_stages": [],
            "source_fragments": [],
            "source_token_indices": [],
            "overwrite_count": 0,
            "was_appended": False,
            "lock_state_relevant": None,
        },
    )
    return entry


def snapshot_tracked_fields(row: RawRowRecord) -> dict[str, str | None]:
    ensure_row_provenance(row)
    return {field: normalize_space(getattr(row, field) or "") or None for field in TRACKED_FIELDS}


def record_stage_diff(
    row: RawRowRecord,
    stage: str,
    before: dict[str, str | None],
    *,
    source_fragments: list[str] | None = None,
    source_token_indices: list[int] | None = None,
    lock_state_relevant: bool | None = None,
) -> None:
    for field in TRACKED_FIELDS:
        prev = before.get(field)
        cur = normalize_space(getattr(row, field) or "") or None
        if prev == cur:
            continue
        entry = _field_entry(row, field)
        if stage not in entry["touched_by_stages"]:
            entry["touched_by_stages"].append(stage)
        entry["final_writer_stage"] = stage
        if prev and cur:
            appended = cur.startswith(prev) and len(cur) > len(prev)
            if appended:
                entry["was_appended"] = True
            else:
                entry["overwrite_count"] += 1
        if source_fragments:
            for fragment in source_fragments:
                norm = normalize_space(fragment)
                if norm and norm not in entry["source_fragments"]:
                    entry["source_fragments"].append(norm)
        if source_token_indices:
            for idx in source_token_indices:
                if isinstance(idx, int) and idx not in entry["source_token_indices"]:
                    entry["source_token_indices"].append(idx)
        if lock_state_relevant is not None:
            entry["lock_state_relevant"] = bool(lock_state_relevant)


def mark_row_merge(parent: RawRowRecord, child: RawRowRecord, stage: str) -> None:
    diag = ensure_row_provenance(parent)
    merge = diag["row_merge_provenance"]
    if stage == "row_boundary_engine":
        merge["merged_by_boundary_engine"] = True
    if stage == "final_stitch":
        merge["merged_by_final_stitch"] = True
    parent_id = row_debug_id(parent)
    child_id = row_debug_id(child)
    if parent_id not in merge["parent_row_ids"]:
        merge["parent_row_ids"].append(parent_id)
    if child_id not in merge["child_row_ids"]:
        merge["child_row_ids"].append(child_id)


def set_boundary_adjusted_confidence(row: RawRowRecord, score: float) -> None:
    diag = ensure_row_provenance(row)
    diag["confidence_provenance"]["boundary_adjusted_confidence"] = float(score)


def set_final_operational_confidence(row: RawRowRecord, score: float) -> None:
    diag = ensure_row_provenance(row)
    diag["confidence_provenance"]["final_operational_confidence"] = float(score)
