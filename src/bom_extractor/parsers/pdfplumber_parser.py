from __future__ import annotations

from pathlib import Path

import pdfplumber

from ..models import PageContext, ParserPageResult, RawRowRecord
from ..utils import normalize_space
from ..zoning import zone_page_lines
from .base import BasePageParser


class PdfPlumberTableParser(BasePageParser):
    parser_name = "pdfplumber_table"

    def parse_page(self, pdf_path: Path, page_ctx: PageContext) -> ParserPageResult:
        result = ParserPageResult(parser_name=self.parser_name, page_number=page_ctx.page_number)
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_ctx.page_number - 1]
            settings_candidates = [
                {"vertical_strategy": "lines", "horizontal_strategy": "lines", "intersection_tolerance": 5},
                {"vertical_strategy": "text", "horizontal_strategy": "text", "text_x_tolerance": 2},
            ]
            row_index = 0
            for settings in settings_candidates:
                tables = page.extract_tables(table_settings=settings) or []
                for t_idx, table in enumerate(tables):
                    for raw_row in table or []:
                        if not raw_row:
                            continue
                        cols = [normalize_space(c or "") for c in raw_row]
                        raw_text = normalize_space(" | ".join(c for c in cols if c))
                        if not raw_text:
                            continue
                        row_index += 1
                        result.rows.append(
                            RawRowRecord(
                                source_file=page_ctx.source_file,
                                source_file_hash=page_ctx.source_file_hash,
                                document_id=page_ctx.document_id,
                                page_number=page_ctx.page_number,
                                row_index_on_page=row_index,
                                raw_text=raw_text,
                                extracted_columns=cols,
                                parser_confidence=0.72,
                                parser_name=self.parser_name,
                                metadata={"table_index_on_page": t_idx, "settings": settings},
                            )
                        )
                if result.rows:
                    break

            if not result.rows:
                result.warnings.append("no_tables_found")
                words = page.extract_words() or []
                zones = zone_page_lines(page.height, [w["top"] for w in words])
                layout = page_ctx.layout_metadata or {}
                if "zone_header_cutoff" in layout and "zone_footer_cutoff" in layout:
                    zones = zones.__class__(
                        page_height=page.height,
                        header_cutoff=float(layout["zone_header_cutoff"]),
                        footer_cutoff=float(layout["zone_footer_cutoff"]),
                    )
                by_line: dict[int, list[str]] = {}
                for w in words:
                    if w["top"] <= zones.header_cutoff or w["bottom"] >= zones.footer_cutoff:
                        continue
                    key = round(w["top"] / 3)
                    by_line.setdefault(key, []).append(normalize_space(w["text"]))
                for row_idx, key in enumerate(sorted(by_line), start=1):
                    cols = [c for c in by_line[key] if c]
                    raw_text = normalize_space(" ".join(cols))
                    if raw_text:
                        result.rows.append(
                            RawRowRecord(
                                source_file=page_ctx.source_file,
                                source_file_hash=page_ctx.source_file_hash,
                                document_id=page_ctx.document_id,
                                page_number=page_ctx.page_number,
                                row_index_on_page=row_idx,
                                raw_text=raw_text,
                                extracted_columns=cols,
                                parser_confidence=0.52,
                                parser_name=self.parser_name,
                            )
                        )

            result.confidence = 0.80 if result.rows else 0.0
            return result
