from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils import normalize_space

LABEL_SYNONYMS = {
    "header_code": ("codice", "code"),
    "header_revision": ("revisione", "revision", "rev"),
    "header_type": ("tipo", "type"),
    "header_description": ("descrizione", "description"),
}

BOUNDARY_STOP_SIGNALS = (
    "specifica",
    "specification",
    "configuration",
    "riferimenti",
    "references",
    "riga",
    "item",
    "qty",
    "q.ty",
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


@dataclass(slots=True)
class HeaderWord:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(slots=True)
class HeaderLabel:
    field: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(slots=True)
class HeaderLine:
    y0: float
    y1: float
    x0: float
    x1: float
    text: str


def _cluster_lines(words: list[HeaderWord], y_quantum: float = 3.0) -> list[HeaderLine]:
    buckets: dict[int, list[HeaderWord]] = {}
    for w in words:
        if normalize_space(w.text):
            buckets.setdefault(round(w.y0 / y_quantum), []).append(w)
    lines: list[HeaderLine] = []
    for key in sorted(buckets):
        row = sorted(buckets[key], key=lambda w: w.x0)
        text = normalize_space(" ".join(w.text for w in row))
        if not text:
            continue
        lines.append(
            HeaderLine(
                y0=min(w.y0 for w in row),
                y1=max(w.y1 for w in row),
                x0=min(w.x0 for w in row),
                x1=max(w.x1 for w in row),
                text=text,
            )
        )
    return lines


def _find_labels(words: list[HeaderWord]) -> list[HeaderLabel]:
    labels: list[HeaderLabel] = []
    for w in words:
        clean = normalize_space(w.text)
        if not clean:
            continue
        tokens = _tokenize(clean)
        if not tokens:
            continue
        for field, synonyms in LABEL_SYNONYMS.items():
            if any(token in synonyms or any(token.startswith(s) for s in synonyms) for token in tokens):
                labels.append(HeaderLabel(field=field, x0=w.x0, y0=w.y0, x1=w.x1, y1=w.y1))
                break
    return labels


def _table_header_anchor(lines: list[HeaderLine]) -> float | None:
    for line in lines:
        lowered = line.text.lower()
        signals = sum(k in lowered for k in ("riga", "item", "codice", "code", "qty", "description", "descrizione"))
        if signals >= 3:
            return line.y0
    return None


def _boundary_stop_anchor(lines: list[HeaderLine], top_hint: float) -> float | None:
    for line in lines:
        if line.y0 <= top_hint:
            continue
        lowered = line.text.lower()
        if any(sig in lowered for sig in BOUNDARY_STOP_SIGNALS):
            return line.y0
    return None


def _extract_value_from_neighbors(field: str, label: HeaderLabel, words: list[HeaderWord], labels: list[HeaderLabel]) -> tuple[str | None, int]:
    same_line = [
        w
        for w in words
        if abs(((w.y0 + w.y1) / 2.0) - ((label.y0 + label.y1) / 2.0)) <= 4.0 and w.x0 >= label.x1 - 1.0
    ]
    next_labels_same_line = [
        l for l in labels
        if l.x0 > label.x1 and abs(((l.y0 + l.y1) / 2.0) - ((label.y0 + label.y1) / 2.0)) <= 5.0
    ]
    stop_x = min((l.x0 for l in next_labels_same_line), default=float("inf"))
    below_line: list[HeaderWord] = []
    if field == "header_description":
        below_line = [
            w
            for w in words
            if w.y0 >= label.y1 and (w.y0 - label.y1) <= 12.0 and abs(w.x0 - label.x0) <= 220.0
        ]
    same_line = [w for w in same_line if w.x0 < stop_x - 1.0 or field == "header_description"]
    candidates = sorted(same_line, key=lambda w: w.x0) + sorted(below_line, key=lambda w: (w.y0, w.x0))
    if not candidates:
        return None, 0

    fragments: list[str] = []
    for w in candidates:
        token = normalize_space(w.text)
        if not token:
            continue
        if token in ("/", ":", "-", "|"):
            continue
        low = token.lower()
        if any(low in synonyms for synonyms in LABEL_SYNONYMS.values()):
            continue
        fragments.append(token)

    if not fragments:
        return None, 0

    if field == "header_revision":
        for frag in fragments:
            m = re.search(r"\b([A-Z0-9]{1,6})\b", frag, re.IGNORECASE)
            if m:
                return m.group(1), 1
        return None, 0

    if field == "header_code":
        joined = " ".join(fragments)
        m = re.search(r"\b([A-Z0-9][A-Z0-9_\-\./]{4,})\b", joined, re.IGNORECASE)
        if m:
            return normalize_space(m.group(1)), 1
        return None, 0

    if field == "header_type":
        stop_terms = {"ultima", "last", "rev", "revision", "revisione"}
        trimmed: list[str] = []
        for frag in fragments:
            if trimmed and frag.lower().strip('.:/') in stop_terms:
                break
            trimmed.append(frag)
        value = normalize_space(" ".join(trimmed[:10]))
        return value or None, 1

    value = normalize_space(" ".join(fragments[:12]))
    return value or None, 1


def extract_targeted_header_fields(
    header_lines: list[str],
    *,
    words: list[tuple[float, float, float, float, str]] | None = None,
    page_height: float | None = None,
) -> dict[str, str | list[str] | tuple[float, float, float, float] | float | int | None]:
    cleaned = [normalize_space(line) for line in header_lines if normalize_space(line)]
    result: dict[str, str | list[str] | tuple[float, float, float, float] | float | int | None] = {
        "header_code": None,
        "header_revision": None,
        "header_type": None,
        "header_description": None,
        "header_raw_lines": cleaned,
        "header_bbox": None,
        "header_zone_confidence": 0.0,
        "header_fields_detected": 0,
        "header_label_matches": 0,
        "header_boundary_conflicts": 0,
        "header_confidence_score": 0.0,
        "warnings": [],
    }

    if not words:
        # Backward-compatible lexical fallback.
        for line in cleaned:
            low = line.lower()
            if result["header_code"] is None:
                m = re.search(r"\b(?:code|codice)\s*[:\-]?\s*([A-Z0-9][A-Z0-9_\-/\.]{4,})\b", low, re.IGNORECASE)
                if m:
                    result["header_code"] = normalize_space(m.group(1).upper())
            if result["header_revision"] is None:
                m = re.search(r"\b(?:rev(?:ision)?|revisione)\s*[:\-]?\s*([A-Z0-9]{1,6})\b", low, re.IGNORECASE)
                if m:
                    result["header_revision"] = normalize_space(m.group(1).upper())
            if result["header_type"] is None:
                m = re.search(r"\b(?:type|tipo)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
                if m:
                    result["header_type"] = normalize_space(m.group(1))
            if result["header_description"] is None:
                m = re.search(r"\b(?:description|descrizione)\s*[:\-]?\s*(.+)$", line, re.IGNORECASE)
                if m:
                    result["header_description"] = normalize_space(m.group(1))
        fields = sum(bool(result[k]) for k in ("header_code", "header_revision", "header_type", "header_description"))
        result["header_fields_detected"] = fields
        result["header_confidence_score"] = round(fields / 4.0, 4)
        result["header_zone_confidence"] = result["header_confidence_score"]
        if fields < 4:
            result["warnings"] = ["header_label_missing"]
        return result

    header_words = [HeaderWord(*w) for w in words if normalize_space(w[4])]
    all_lines = _cluster_lines(header_words)
    table_anchor = _table_header_anchor(all_lines)
    top_hint = min((line.y0 for line in all_lines), default=0.0)
    stop_anchor = _boundary_stop_anchor(all_lines, top_hint)

    upper_limit = page_height * 0.35 if page_height else float("inf")
    if table_anchor is not None:
        upper_limit = min(upper_limit, table_anchor)
    if stop_anchor is not None:
        upper_limit = min(upper_limit, stop_anchor)

    zone_words = [w for w in header_words if w.y1 <= upper_limit + 1.0]
    labels = _find_labels(zone_words)

    for field in ("header_code", "header_revision", "header_type", "header_description"):
        anchors = [lbl for lbl in labels if lbl.field == field]
        if not anchors:
            continue
        best_value: str | None = None
        best_hit = 0
        for anchor in sorted(anchors, key=lambda a: (a.y0, a.x0)):
            value, hit = _extract_value_from_neighbors(field, anchor, zone_words, labels)
            if not value:
                continue
            if field == "header_revision":
                is_preferred = bool(re.search(r"\d", value)) and len(value) <= 6
                if best_value is None:
                    best_value = value
                    best_hit = hit
                elif is_preferred and not (bool(re.search(r"\d", best_value)) and len(best_value) <= 6):
                    best_value = value
                    best_hit = hit
                elif is_preferred and bool(re.search(r"\d", best_value)) and len(value) < len(best_value):
                    best_value = value
                    best_hit = hit
                continue
            if best_value is None or len(value) > len(best_value):
                best_value = value
                best_hit = hit
        result[field] = best_value
        result["header_label_matches"] = int(result["header_label_matches"] or 0) + len(anchors)
        if best_hit == 0:
            result["warnings"].append("header_value_ambiguous")

    raw_lines = [line.text for line in all_lines if line.y1 <= upper_limit + 1.0]
    result["header_raw_lines"] = raw_lines

    if zone_words:
        result["header_bbox"] = (
            round(min(w.x0 for w in zone_words), 2),
            round(min(w.y0 for w in zone_words), 2),
            round(max(w.x1 for w in zone_words), 2),
            round(max(w.y1 for w in zone_words), 2),
        )

    fields_detected = sum(bool(result[k]) for k in ("header_code", "header_revision", "header_type", "header_description"))
    result["header_fields_detected"] = fields_detected

    boundary_conflicts = 0
    if table_anchor is None and stop_anchor is None:
        boundary_conflicts += 1
        result["warnings"].append("header_boundary_uncertain")
    if table_anchor is not None and zone_words and any(w.y0 >= table_anchor - 1.0 for w in zone_words):
        boundary_conflicts += 1
        result["warnings"].append("header_zone_overlap_with_table")
    result["header_boundary_conflicts"] = boundary_conflicts

    if fields_detected < 4:
        result["warnings"].append("header_label_missing")

    label_score = min(1.0, float(result["header_label_matches"] or 0) / 4.0)
    boundary_penalty = min(0.6, boundary_conflicts * 0.25)
    confidence = max(0.0, (fields_detected / 4.0) * 0.7 + label_score * 0.3 - boundary_penalty)
    result["header_confidence_score"] = round(confidence, 4)
    result["header_zone_confidence"] = round(max(0.0, confidence - 0.05 * boundary_conflicts), 4)
    result["warnings"] = sorted(set(result["warnings"]))

    return result
