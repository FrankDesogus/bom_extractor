from __future__ import annotations

import re

from ..utils import normalize_space

CODE_PATTERNS = [
    re.compile(r"\b(?:code|codice)\s*[:\-]?\s*([A-Z0-9][A-Z0-9_\-/\.]{4,})\b", re.IGNORECASE),
]
REV_PATTERNS = [
    re.compile(r"\b(?:rev(?:ision)?|revisione)\s*[:\-]?\s*([A-Z0-9]{1,6})\b", re.IGNORECASE),
]
TYPE_PATTERNS = [
    re.compile(r"\b(?:type|tipo)\s*[:\-]?\s*([^|]{2,40})", re.IGNORECASE),
]
DESCRIPTION_PATTERNS = [
    re.compile(r"\b(?:description|descrizione)\s*[:\-]?\s*(.+)$", re.IGNORECASE),
]


def _extract_first(lines: list[str], patterns: list[re.Pattern[str]]) -> str | None:
    for line in lines:
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                return normalize_space(match.group(1))
    return None


def extract_targeted_header_fields(header_lines: list[str]) -> dict[str, str | list[str] | None]:
    cleaned = [normalize_space(line) for line in header_lines if normalize_space(line)]
    return {
        "header_code": _extract_first(cleaned, CODE_PATTERNS),
        "header_revision": _extract_first(cleaned, REV_PATTERNS),
        "header_type": _extract_first(cleaned, TYPE_PATTERNS),
        "header_description": _extract_first(cleaned, DESCRIPTION_PATTERNS),
        "header_raw_lines": cleaned,
    }
