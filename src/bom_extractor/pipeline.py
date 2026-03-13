from __future__ import annotations

import logging
from pathlib import Path

import fitz

from .config import ExtractionConfig
from .logging_utils import configure_logging
from .models import DocumentSummary, PageContext, RawRowRecord
from .normalizer import stitch_multiline_rows, weak_map_columns
from .storage import StorageManager
from .utils import looks_like_item, normalize_space, sha256_file
from .validators import validate_row
from .parsers.camelot_parser import CamelotLatticeParser
from .parsers.ocr_parser import OCRFallbackParser
from .parsers.pdfplumber_parser import PdfPlumberTableParser
from .parsers.pymupdf_parser import PyMuPDFWordsParser


class ExtractionPipeline:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = configure_logging(config.output_dir)
        self.storage = StorageManager(config.output_dir)
        self.parsers = self._build_parsers()

    def _build_parsers(self):
        parsers = []
        if self.config.enable_camelot:
            parsers.append(CamelotLatticeParser())
        if self.config.enable_pdfplumber:
            parsers.append(PdfPlumberTableParser())
        if self.config.enable_pymupdf:
            parsers.append(PyMuPDFWordsParser())
        if self.config.enable_ocr:
            parsers.append(OCRFallbackParser())
        return parsers

    def parse_input(self, input_path: Path) -> list[RawRowRecord]:
        pdfs = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
        all_rows: list[RawRowRecord] = []
        summaries: list[DocumentSummary] = []
        for pdf_path in pdfs:
            rows, summary = self.parse_document(pdf_path)
            all_rows.extend(rows)
            summaries.append(summary)
        combined = DocumentSummary(
            source_file=str(input_path),
            source_file_hash="",
            document_id="batch",
            pages_seen=sum(s.pages_seen for s in summaries),
            rows_emitted=sum(s.rows_emitted for s in summaries),
            parser_usage=self._merge_parser_usage(summaries),
            warnings=[w for s in summaries for w in s.warnings],
            errors=[e for s in summaries for e in s.errors],
        )
        self.storage.write_jsonl(all_rows)
        if self.config.write_csv:
            self.storage.write_csv(all_rows)
        if self.config.write_parquet:
            self.storage.write_parquet(all_rows)
        self.storage.write_summary(combined)
        return all_rows

    def parse_document(self, pdf_path: Path) -> tuple[list[RawRowRecord], DocumentSummary]:
        file_hash = sha256_file(pdf_path)
        document_id = f"{pdf_path.stem}:{file_hash[:12]}"
        summary = DocumentSummary(
            source_file=str(pdf_path),
            source_file_hash=file_hash,
            document_id=document_id,
        )
        rows: list[RawRowRecord] = []
        doc = fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            limit = min(total_pages, self.config.max_pages) if self.config.max_pages else total_pages
            for page_idx in range(limit):
                page = doc[page_idx]
                page_text = normalize_space(page.get_text("text"))
                page_ctx = PageContext(
                    source_file=str(pdf_path),
                    source_file_hash=file_hash,
                    document_id=document_id,
                    page_number=page_idx + 1,
                    page_rotation=page.rotation,
                    raw_page_text=page_text,
                )
                page_rows = self._parse_page(pdf_path, page_ctx)
                rows.extend(page_rows)
                if page_rows:
                    chosen_parser = page_rows[0].parser_name
                    summary.parser_usage[chosen_parser] = summary.parser_usage.get(chosen_parser, 0) + 1
                summary.pages_seen += 1
            summary.rows_emitted = len(rows)
            return rows, summary
        except Exception as exc:
            summary.errors.append(f"document_error:{type(exc).__name__}:{exc}")
            self.logger.error(
                "document parse failed",
                extra={"structured": {"source_file": str(pdf_path), "error": str(exc)}},
            )
            if not self.config.continue_on_error:
                raise
            return rows, summary
        finally:
            doc.close()

    def _parse_page(self, pdf_path: Path, page_ctx: PageContext) -> list[RawRowRecord]:
        parser_results = []
        for parser in self.parsers:
            try:
                result = parser.parse_page(pdf_path, page_ctx)
                parser_results.append(result)
                self.logger.info(
                    "parser completed",
                    extra={"structured": {
                        "source_file": page_ctx.source_file,
                        "page_number": page_ctx.page_number,
                        "parser_name": result.parser_name,
                        "rows": len(result.rows),
                        "confidence": result.confidence,
                        "warnings": result.warnings,
                        "errors": result.errors,
                    }},
                )
            except Exception as exc:
                self.logger.error(
                    "parser failed",
                    extra={"structured": {
                        "source_file": page_ctx.source_file,
                        "page_number": page_ctx.page_number,
                        "parser_name": getattr(parser, "parser_name", parser.__class__.__name__),
                        "error": f"{type(exc).__name__}:{exc}",
                    }},
                )
                if not self.config.continue_on_error:
                    raise

        best = self._choose_best_result(parser_results, page_ctx)
        normalized: list[RawRowRecord] = []
        for row in best.rows:
            row = weak_map_columns(row)
            row = validate_row(row)
            normalized.append(row)
        normalized = stitch_multiline_rows(normalized)
        return normalized

    def _choose_best_result(self, parser_results, page_ctx: PageContext):
        if not parser_results:
            raise RuntimeError("No parser results available")

        def score(res):
            item_like = 0
            for row in res.rows:
                first = row.extracted_columns[0] if row.extracted_columns else None
                if looks_like_item(first):
                    item_like += 1
            item_ratio = item_like / len(res.rows) if res.rows else 0.0
            suspicious_single_blob = 1 if len(res.rows) == 1 and item_like == 0 else 0
            return (
                item_like,
                item_ratio,
                res.confidence,
                len(res.rows),
                -suspicious_single_blob,
                -len(res.errors),
            )

        best = max(parser_results, key=score)
        if best.confidence < self.config.low_confidence_threshold:
            self.logger.warning(
                "low confidence page parse",
                extra={"structured": {
                    "source_file": page_ctx.source_file,
                    "page_number": page_ctx.page_number,
                    "parser_name": best.parser_name,
                    "confidence": best.confidence,
                }},
            )
        return best

    @staticmethod
    def _merge_parser_usage(summaries: list[DocumentSummary]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for summary in summaries:
            for key, value in summary.parser_usage.items():
                merged[key] = merged.get(key, 0) + value
        return merged
