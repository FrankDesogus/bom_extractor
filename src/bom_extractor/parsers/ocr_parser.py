from __future__ import annotations

from pathlib import Path

from ..models import PageContext, ParserPageResult
from .base import BasePageParser


class OCRFallbackParser(BasePageParser):
    parser_name = "ocr_fallback"

    def parse_page(self, pdf_path: Path, page_ctx: PageContext) -> ParserPageResult:
        result = ParserPageResult(parser_name=self.parser_name, page_number=page_ctx.page_number)
        result.warnings.append("ocr_not_implemented_in_phase1")
        result.confidence = 0.0
        return result
