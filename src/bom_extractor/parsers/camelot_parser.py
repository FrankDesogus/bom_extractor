from __future__ import annotations

from pathlib import Path

from ..models import PageContext, ParserPageResult, RawRowRecord
from ..utils import normalize_space
from .base import BasePageParser

try:
    import camelot
except Exception:  # pragma: no cover
    camelot = None


class CamelotLatticeParser(BasePageParser):
    parser_name = "camelot_lattice"

    def parse_page(self, pdf_path: Path, page_ctx: PageContext) -> ParserPageResult:
        result = ParserPageResult(parser_name=self.parser_name, page_number=page_ctx.page_number)
        if camelot is None:
            result.errors.append("camelot_not_available")
            return result

        tables = None
        used_flavor = "lattice"
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(str(pdf_path), pages=str(page_ctx.page_number), flavor=flavor)
                used_flavor = flavor
            except Exception as exc:
                result.errors.append(f"camelot_{flavor}_error:{type(exc).__name__}:{exc}")
                continue
            if tables:
                break

        if not tables:
            result.warnings.append("no_tables_found")
            return result

        row_index = 0
        for t_idx, table in enumerate(tables):
            df = table.df
            for _, series in df.iterrows():
                cols = [normalize_space(str(v)) for v in series.tolist()]
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
                        parser_confidence=0.84,
                        parser_name=self.parser_name,
                        metadata={
                            "table_index_on_page": t_idx,
                            "camelot_flavor": used_flavor,
                            "camelot_accuracy": getattr(table, "accuracy", None),
                        },
                    )
                )
        result.confidence = 0.86 if result.rows else 0.0
        return result
