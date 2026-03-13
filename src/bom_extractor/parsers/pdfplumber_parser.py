from __future__ import annotations

from pathlib import Path

import pdfplumber

from ..models import PageContext, ParserPageResult, RawRowRecord
from ..utils import normalize_space
from .base import BasePageParser


class PdfPlumberTableParser(BasePageParser):
    parser_name = "pdfplumber_table"

    def parse_page(self, pdf_path: Path, page_ctx: PageContext) -> ParserPageResult:
        result = ParserPageResult(parser_name=self.parser_name, page_number=page_ctx.page_number)
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_ctx.page_number - 1]
            settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }
            tables = page.extract_tables(table_settings=settings) or []
            row_index = 0
            for t_idx, table in enumerate(tables):
                for raw_row in table or []:
                    if not raw_row:
                        continue
                    row_index += 1
                    cols = [normalize_space(c or "") for c in raw_row]
                    raw_text = normalize_space(" | ".join(c for c in cols if c))
                    if not raw_text:
                        continue
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
                            metadata={"table_index_on_page": t_idx},
                        )
                    )
            if not result.rows:
                result.warnings.append("no_tables_found")
                page_text = normalize_space(page.extract_text() or "")
                if page_text:
                    result.metadata["raw_page_text_preview"] = page_text[:500]
            result.confidence = 0.78 if result.rows else 0.0
            return result
