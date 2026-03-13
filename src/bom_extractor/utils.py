from __future__ import annotations

import hashlib
import re
from pathlib import Path


HEADER_PATTERNS = [
    re.compile(r"\briga\b.*\bitem\b", re.IGNORECASE),
    re.compile(r"\btipo\b.*\btype\b", re.IGNORECASE),
    re.compile(r"\bcodice\b.*\bcode\b", re.IGNORECASE),
]

FOOTER_PATTERNS = [
    re.compile(r"proprietary information", re.IGNORECASE),
    re.compile(r"documento emesso", re.IGNORECASE),
    re.compile(r"pagina\s*/\s*sheet", re.IGNORECASE),
]

ITEM_PATTERN = re.compile(r"^\s*\d{3,5}\s*$")
CODE_PATTERN = re.compile(r"[A-Z]?\d{6,}")
QUANTITY_PATTERN = re.compile(r"^-?\d+(?:[\.,]\d+)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def looks_like_header(text: str) -> bool:
    text = normalize_space(text)
    matches = sum(1 for p in HEADER_PATTERNS if p.search(text))
    return matches >= 2


def looks_like_footer(text: str) -> bool:
    text = normalize_space(text)
    return any(p.search(text) for p in FOOTER_PATTERNS)


def looks_like_item(value: str | None) -> bool:
    if not value:
        return False
    return bool(ITEM_PATTERN.match(normalize_space(value)))


def looks_like_code(value: str | None) -> bool:
    if not value:
        return False
    return bool(CODE_PATTERN.search(normalize_space(value)))


def looks_like_quantity(value: str | None) -> bool:
    if not value:
        return False
    return bool(QUANTITY_PATTERN.match(normalize_space(value)))
