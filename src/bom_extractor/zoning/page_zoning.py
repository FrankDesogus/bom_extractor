from __future__ import annotations

from dataclasses import dataclass

from ..utils import looks_like_footer, looks_like_header, looks_like_item, looks_like_quantity, normalize_space


@dataclass(slots=True)
class PageZones:
    page_height: float
    header_cutoff: float
    footer_cutoff: float


@dataclass(slots=True)
class PageLayout:
    zones: PageZones
    header_lines: list[str]
    table_lines: list[str]
    footer_lines: list[str]
    background_noise_lines: list[str]
    warnings: list[str]
    confidence: float


def zone_page_lines(page_height: float, line_ys: list[float]) -> PageZones:
    if not line_ys:
        return PageZones(page_height=page_height, header_cutoff=page_height * 0.08, footer_cutoff=page_height * 0.92)

    y_min = min(line_ys)
    y_max = max(line_ys)
    dynamic_top = max(page_height * 0.05, y_min + page_height * 0.02)
    dynamic_bottom = min(page_height * 0.97, y_max - page_height * 0.02)
    return PageZones(page_height=page_height, header_cutoff=dynamic_top, footer_cutoff=dynamic_bottom)


def infer_page_layout(page_height: float, words: list[tuple[float, float, float, float, str]]) -> PageLayout:
    if not words:
        zones = zone_page_lines(page_height, [])
        return PageLayout(
            zones=zones,
            header_lines=[],
            table_lines=[],
            footer_lines=[],
            background_noise_lines=[],
            warnings=["no_words_for_layout", "low_layout_confidence"],
            confidence=0.0,
        )

    by_line: dict[int, list[tuple[float, float, float, float, str]]] = {}
    for x0, y0, x1, y1, text in words:
        norm = normalize_space(text)
        if norm:
            by_line.setdefault(round(y0 / 3), []).append((x0, y0, x1, y1, norm))

    line_records: list[tuple[float, float, str, int]] = []
    table_like_indices: list[int] = []
    for idx, key in enumerate(sorted(by_line.keys())):
        cells = sorted(by_line[key], key=lambda c: c[0])
        line_top = min(c[1] for c in cells)
        line_bottom = max(c[3] for c in cells)
        line_text = normalize_space(" ".join(c[4] for c in cells))
        if not line_text:
            continue
        token_count = len(line_text.split(" "))
        first_token = line_text.split(" ")[0]
        has_item_anchor = looks_like_item(first_token)
        has_quantity_signal = any(looks_like_quantity(tok) for tok in line_text.split(" "))
        in_bottom_band = line_top > page_height * 0.85
        if not in_bottom_band and not looks_like_footer(line_text) and (has_item_anchor or has_quantity_signal or token_count >= 4):
            table_like_indices.append(idx)
        line_records.append((line_top, line_bottom, line_text, token_count))

    zones = zone_page_lines(page_height, [rec[0] for rec in line_records])
    warnings: list[str] = []
    if line_records and table_like_indices:
        first_table_line = line_records[min(table_like_indices)][0]
        last_table_line = line_records[max(table_like_indices)][1]
        zones = PageZones(
            page_height=page_height,
            header_cutoff=max(page_height * 0.04, first_table_line - page_height * 0.02),
            footer_cutoff=min(page_height * 0.98, last_table_line + page_height * 0.02),
        )
    else:
        warnings.extend(["table_zone_inference_weak", "low_layout_confidence"])

    header_lines: list[str] = []
    table_lines: list[str] = []
    footer_lines: list[str] = []
    background_noise_lines: list[str] = []
    for top, bottom, text, token_count in line_records:
        if token_count <= 1 and len(text) <= 2:
            background_noise_lines.append(text)
            continue
        if bottom <= zones.header_cutoff or looks_like_header(text):
            header_lines.append(text)
        elif top >= zones.footer_cutoff or looks_like_footer(text):
            footer_lines.append(text)
        else:
            table_lines.append(text)

    confidence = min(1.0, max(0.2, len(table_lines) / max(1, len(line_records))))
    if confidence < 0.45 and "low_layout_confidence" not in warnings:
        warnings.append("low_layout_confidence")

    return PageLayout(
        zones=zones,
        header_lines=header_lines,
        table_lines=table_lines,
        footer_lines=footer_lines,
        background_noise_lines=background_noise_lines,
        warnings=warnings,
        confidence=round(confidence, 4),
    )
