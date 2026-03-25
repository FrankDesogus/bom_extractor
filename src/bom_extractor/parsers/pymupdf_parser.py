from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import fitz

from ..models import PageContext, ParserPageResult, RawRowRecord
from ..utils import normalize_space
from ..zoning import zone_page_lines
from .base import BasePageParser


class PyMuPDFWordsParser(BasePageParser):
    parser_name = "pymupdf_words"
    _ITEM_ANCHOR_RE = re.compile(r"^\d{3,4}$")
    _CODE_ANCHOR_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-./]{2,}$")
    _REV_ANCHOR_RE = re.compile(r"^(?:[A-Z]|[A-Z]\d|\d{1,2}|REV\.?)$", re.IGNORECASE)
    _X_CLUSTER_GAP = 18.0
    _X_ALIGN_TOLERANCE = 14.0

    @classmethod
    def _cluster_x_positions(cls, xs: list[float], tolerance: float) -> list[float]:
        if not xs:
            return []
        centers: list[float] = []
        current_cluster: list[float] = [xs[0]]
        for x in xs[1:]:
            if abs(x - current_cluster[-1]) <= tolerance:
                current_cluster.append(x)
                continue
            centers.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [x]
        centers.append(sum(current_cluster) / len(current_cluster))
        return centers

    @classmethod
    def _looks_like_table_row_anchor(cls, cells: list[tuple]) -> bool:
        tokens = [normalize_space(c[4]) for c in sorted(cells, key=lambda x: x[0])]
        tokens = [t for t in tokens if t]
        if len(tokens) < 3:
            return False
        has_item = any(cls._ITEM_ANCHOR_RE.match(t) for t in tokens)
        has_code = any(cls._CODE_ANCHOR_RE.match(t) and any(ch.isdigit() for ch in t) for t in tokens)
        has_rev = any(cls._REV_ANCHOR_RE.match(t) for t in tokens[-3:])
        return has_item and has_code and has_rev

    @classmethod
    def _looks_like_clustered_table_row(cls, cells: list[tuple], x_hints: list[float]) -> bool:
        if len(cells) < 3 or len(x_hints) < 6:
            return False
        sorted_cells = sorted(cells, key=lambda x: x[0])
        cell_xs = [float(c[0]) for c in sorted_cells]
        candidate_clusters = cls._cluster_x_positions(cell_xs, cls._X_CLUSTER_GAP)
        if len(candidate_clusters) < 3:
            return False

        span_min = min(float(c[0]) for c in sorted_cells)
        span_max = max(float(c[2]) for c in sorted_cells)
        span_width = span_max - span_min
        if span_width <= 0:
            return False
        total_word_width = sum(max(0.0, float(c[2]) - float(c[0])) for c in sorted_cells)
        coverage_ratio = total_word_width / span_width
        if coverage_ratio > 0.68:
            return False

        reference_clusters = cls._cluster_x_positions(sorted(float(x) for x in x_hints), cls._X_CLUSTER_GAP)
        if len(reference_clusters) < 3:
            return False

        aligned = 0
        for cluster_x in candidate_clusters:
            if any(abs(cluster_x - ref_x) <= cls._X_ALIGN_TOLERANCE for ref_x in reference_clusters):
                aligned += 1
        return aligned >= 3 or (aligned >= 2 and len(candidate_clusters) <= 4)

    def parse_page(self, pdf_path: Path, page_ctx: PageContext) -> ParserPageResult:
        result = ParserPageResult(parser_name=self.parser_name, page_number=page_ctx.page_number)
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_ctx.page_number - 1]
            words = page.get_text("words")
            if not words:
                result.warnings.append("no_words_found")
                return result

            layout = page_ctx.layout_metadata or {}
            zones = zone_page_lines(page.rect.height, [w[1] for w in words])
            if "zone_header_cutoff" in layout and "zone_footer_cutoff" in layout:
                zones = zones.__class__(
                    page_height=page.rect.height,
                    header_cutoff=float(layout["zone_header_cutoff"]),
                    footer_cutoff=float(layout["zone_footer_cutoff"]),
                )
            raw_buckets: dict[int, list[tuple]] = defaultdict(list)
            for w in words:
                x0, y0, x1, y1, text, *_ = w
                raw_buckets[round(y0 / 3)].append((x0, y0, x1, y1, text))

            buckets: dict[int, list[tuple]] = defaultdict(list)
            skipped_header_footer = 0
            x_hints: list[float] = []
            for bucket_key, cells in raw_buckets.items():
                sorted_cells = sorted(cells, key=lambda x: x[0])
                keep_footer_bucket = False
                if sorted_cells:
                    in_footer_band = any(c[3] >= zones.footer_cutoff for c in sorted_cells)
                    if in_footer_band:
                        keep_footer_bucket = self._looks_like_table_row_anchor(sorted_cells) or self._looks_like_clustered_table_row(
                            sorted_cells, x_hints
                        )

                for cell in sorted_cells:
                    x0, y0, x1, y1, _ = cell
                    if y0 <= zones.header_cutoff or (y1 >= zones.footer_cutoff and not keep_footer_bucket):
                        skipped_header_footer += 1
                        continue
                    buckets[bucket_key].append(cell)
                    x_hints.append(float(x0))

            rows: list[RawRowRecord] = []
            for row_idx, bucket_key in enumerate(sorted(buckets), start=1):
                cells = sorted(buckets[bucket_key], key=lambda x: x[0])
                texts = [normalize_space(c[4]) for c in cells if normalize_space(c[4])]
                raw_text = normalize_space(" ".join(texts))
                if not raw_text:
                    continue
                bbox = (
                    min(c[0] for c in cells),
                    min(c[1] for c in cells),
                    max(c[2] for c in cells),
                    max(c[3] for c in cells),
                )
                rows.append(
                    RawRowRecord(
                        source_file=page_ctx.source_file,
                        source_file_hash=page_ctx.source_file_hash,
                        document_id=page_ctx.document_id,
                        page_number=page_ctx.page_number,
                        row_index_on_page=row_idx,
                        raw_text=raw_text,
                        extracted_columns=texts,
                        parser_confidence=0.55,
                        parser_name=self.parser_name,
                        bbox_row=bbox,
                        metadata={"word_boxes": [{"x0": c[0], "y0": c[1], "x1": c[2], "y1": c[3], "text": normalize_space(c[4])} for c in cells]},
                    )
                )

            result.rows = rows
            result.metadata.update(
                {
                    "zone_header_cutoff": zones.header_cutoff,
                    "zone_footer_cutoff": zones.footer_cutoff,
                    "header_footer_words_skipped": skipped_header_footer,
                    "column_x_hints": sorted(x_hints),
                    "column_count_hint": max((len(r.extracted_columns) for r in rows), default=0),
                }
            )
            result.confidence = 0.75 if rows else 0.0
            return result
        finally:
            doc.close()
