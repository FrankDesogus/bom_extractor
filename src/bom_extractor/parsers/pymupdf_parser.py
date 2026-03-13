from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import fitz

from ..models import PageContext, ParserPageResult, RawRowRecord
from ..utils import normalize_space
from ..zoning import zone_page_lines
from .base import BasePageParser


class PyMuPDFWordsParser(BasePageParser):
    parser_name = "pymupdf_words"

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
            buckets: dict[int, list[tuple]] = defaultdict(list)
            skipped_header_footer = 0
            for w in words:
                x0, y0, x1, y1, text, *_ = w
                if y0 <= zones.header_cutoff or y1 >= zones.footer_cutoff:
                    skipped_header_footer += 1
                    continue
                buckets[round(y0 / 3)].append((x0, y0, x1, y1, text))

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
                    )
                )

            result.rows = rows
            result.metadata.update(
                {
                    "zone_header_cutoff": zones.header_cutoff,
                    "zone_footer_cutoff": zones.footer_cutoff,
                    "header_footer_words_skipped": skipped_header_footer,
                }
            )
            result.confidence = 0.75 if rows else 0.0
            return result
        finally:
            doc.close()
