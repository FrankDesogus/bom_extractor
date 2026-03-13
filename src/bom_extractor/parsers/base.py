from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import PageContext, ParserPageResult


class BasePageParser(ABC):
    parser_name: str

    @abstractmethod
    def parse_page(self, pdf_path: Path, page_ctx: PageContext) -> ParserPageResult:
        raise NotImplementedError
