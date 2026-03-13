from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PageZones:
    page_height: float
    header_cutoff: float
    footer_cutoff: float


def zone_page_lines(page_height: float, line_ys: list[float]) -> PageZones:
    """Infer conservative header/footer zones from y-distribution."""
    if not line_ys:
        return PageZones(page_height=page_height, header_cutoff=page_height * 0.08, footer_cutoff=page_height * 0.92)

    y_min = min(line_ys)
    y_max = max(line_ys)
    dynamic_top = max(page_height * 0.06, y_min + page_height * 0.02)
    dynamic_bottom = min(page_height * 0.95, y_max - page_height * 0.02)
    return PageZones(page_height=page_height, header_cutoff=dynamic_top, footer_cutoff=dynamic_bottom)
