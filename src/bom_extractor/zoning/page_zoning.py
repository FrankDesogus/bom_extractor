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
    warnings: list[str]


def zone_page_lines(page_height: float, line_ys: list[float]) -> PageZones:
    """Infer conservative header/footer zones from y-distribution."""
    if not line_ys:
        return PageZones(page_height=page_height, header_cutoff=page_height * 0.08, footer_cutoff=page_height * 0.92)

    y_min = min(line_ys)
    y_max = max(line_ys)
    dynamic_top = max(page_height * 0.06, y_min + page_height * 0.02)
    dynamic_bottom = min(page_height * 0.95, y_max - page_height * 0.02)
    return PageZones(page_height=page_height, header_cutoff=dynamic_top, footer_cutoff=dynamic_bottom)


def infer_page_layout(page_height: float, words: list[tuple[float, float, float, float, str]]) -> PageLayout:
    """Infer header/table/footer zones while preserving text evidence from every zone."""
    if not words:
        zones = zone_page_lines(page_height, [])
        return PageLayout(zones=zones, header_lines=[], table_lines=[], footer_lines=[], warnings=["no_words_for_layout"])

    by_line: dict[int, list[tuple[float, float, float, float, str]]] = {}
    for word in words:
        x0, y0, x1, y1, text = word
        by_line.setdefault(round(y0 / 3), []).append((x0, y0, x1, y1, text))

    line_records: list[tuple[float, float, str]] = []
    table_like_indices: list[int] = []
    for idx, key in enumerate(sorted(by_line.keys())):
        cells = sorted(by_line[key], key=lambda c: c[0])
        line_top = min(c[1] for c in cells)
        line_bottom = max(c[3] for c in cells)
        line_text = normalize_space(" ".join(normalize_space(c[4]) for c in cells if normalize_space(c[4])))
        if not line_text:
            continue
        first_token = line_text.split(" ")[0]
        has_item_anchor = looks_like_item(first_token)
        has_quantity_signal = any(looks_like_quantity(tok) for tok in line_text.split(" "))
        if has_item_anchor or (has_quantity_signal and "|" in line_text):
            table_like_indices.append(idx)
        line_records.append((line_top, line_bottom, line_text))

    zones = zone_page_lines(page_height, [rec[0] for rec in line_records])
    warnings: list[str] = []
    if line_records and table_like_indices:
        first_table_line = line_records[min(table_like_indices)][0]
        last_table_line = line_records[max(table_like_indices)][1]
        zones = PageZones(
            page_height=page_height,
            header_cutoff=max(page_height * 0.05, first_table_line - page_height * 0.015),
            footer_cutoff=min(page_height * 0.97, last_table_line + page_height * 0.02),
        )
    else:
        warnings.append("table_zone_inference_weak")

    header_lines: list[str] = []
    table_lines: list[str] = []
    footer_lines: list[str] = []
    for top, bottom, text in line_records:
        if bottom <= zones.header_cutoff or looks_like_header(text):
            header_lines.append(text)
        elif top >= zones.footer_cutoff or looks_like_footer(text):
            footer_lines.append(text)
        else:
            table_lines.append(text)

    return PageLayout(
        zones=zones,
        header_lines=header_lines,
        table_lines=table_lines,
        footer_lines=footer_lines,
        warnings=warnings,
    )
