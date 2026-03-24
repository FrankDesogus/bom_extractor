from __future__ import annotations

from pathlib import Path

import fitz

from .config import ExtractionConfig
from .fusion import PageResultFuser
from .logging_utils import configure_logging
from .models import DocumentSummary, PageContext, PageOutput, RawRowRecord, RowOutput
from .normalizer import stitch_multiline_rows, weak_map_columns
from .normalization import apply_structure_assisted_reconstruction
from .normalization.header_extraction import extract_targeted_header_fields
from .normalization.row_boundary_engine import apply_row_boundary_engine
from .normalization.row_reconstruction import apply_page_lane_inference, continuation_metrics
from .parsers.camelot_parser import CamelotLatticeParser
from .parsers.ocr_parser import OCRFallbackParser
from .parsers.pdfplumber_parser import PdfPlumberTableParser
from .parsers.pymupdf_parser import PyMuPDFWordsParser
from .storage import StorageManager
from .utils import normalize_space, sha256_file
from .validators import validate_row
from .zoning import infer_page_layout


class ExtractionPipeline:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = configure_logging(config.output_dir)
        self.storage = StorageManager(config.output_dir)
        self.parsers = self._build_parsers()
        self.fuser = PageResultFuser(low_confidence_threshold=config.low_confidence_threshold)

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
            fusion_decisions=[d for s in summaries for d in s.fusion_decisions],
            page_layouts=[p for s in summaries for p in s.page_layouts],
            pages=[p for s in summaries for p in s.pages],
            page_state_distribution=self._merge_counter_maps(summaries, "page_state_distribution"),
            structural_row_metrics=self._merge_metric_maps(summaries),
            document_metadata={"documents": [s.document_metadata for s in summaries]},
        )
        self.storage.write_jsonl(all_rows)
        self.storage.write_provenance_jsonl(all_rows)
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
            document_metadata={"source_file": str(pdf_path), "source_file_hash": file_hash, "document_id": document_id},
        )
        rows: list[RawRowRecord] = []

        doc = fitz.open(pdf_path)
        try:
            total_pages = len(doc)
            limit = min(total_pages, self.config.max_pages) if self.config.max_pages else total_pages
            previous_page_state: str | None = None
            for page_idx in range(limit):
                page = doc[page_idx]
                page_ctx = PageContext(
                    source_file=str(pdf_path),
                    source_file_hash=file_hash,
                    document_id=document_id,
                    page_number=page_idx + 1,
                    page_rotation=page.rotation,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                    raw_page_text=normalize_space(page.get_text("text")),
                )
                page_ctx.layout_metadata = self._build_page_layout(page, page_ctx)
                header_info = extract_targeted_header_fields(
                    page_ctx.layout_metadata.get("header_lines", []),
                    words=page_ctx.layout_metadata.get("page_words", []),
                    page_height=page_ctx.page_height,
                )
                page_output = PageOutput(
                    page_number=page_ctx.page_number,
                    page_state=None,
                    header_code=header_info.get("header_code"),
                    header_revision=header_info.get("header_revision"),
                    header_type=header_info.get("header_type"),
                    header_description=header_info.get("header_description"),
                    header_fields_raw=header_info.get("header_raw_lines", []),
                    header_raw_lines=header_info.get("header_raw_lines", []),
                    header_bbox=header_info.get("header_bbox"),
                    header_zone_confidence=header_info.get("header_zone_confidence"),
                    footer_fields_raw=page_ctx.layout_metadata.get("footer_lines", []),
                    layout_model={
                        **page_ctx.layout_metadata,
                        "header_extraction_metrics": {
                            "header_fields_detected": header_info.get("header_fields_detected", 0),
                            "header_label_matches": header_info.get("header_label_matches", 0),
                            "header_boundary_conflicts": header_info.get("header_boundary_conflicts", 0),
                            "header_confidence_score": header_info.get("header_confidence_score", 0.0),
                        },
                    },
                    warnings=list(page_ctx.layout_metadata.get("layout_warnings", [])) + list(header_info.get("warnings", [])),
                )
                summary.page_layouts.append({
                    "page_number": page_ctx.page_number,
                    "page_state": page_output.page_state,
                    "header_fields_raw": page_output.header_fields_raw,
                    "header_code": page_output.header_code,
                    "header_revision": page_output.header_revision,
                    "header_type": page_output.header_type,
                    "header_description": page_output.header_description,
                    "header_bbox": page_output.header_bbox,
                    "header_zone_confidence": page_output.header_zone_confidence,
                    "header_extraction_metrics": page_output.layout_model.get("header_extraction_metrics", {}),
                    "footer_fields_raw": page_output.footer_fields_raw,
                    "page_warnings": page_output.warnings,
                })
                page_rows = self._parse_page(pdf_path, page_ctx, summary, page_output)
                page_state = self._classify_page_state(page_output, page_rows, previous_page_state)
                page_output.page_state = page_state
                previous_page_state = page_state
                summary.page_state_distribution[page_state] = summary.page_state_distribution.get(page_state, 0) + 1
                self._apply_page_header_policy(page_output)
                if summary.page_layouts:
                    summary.page_layouts[-1]["page_state"] = page_state
                    summary.page_layouts[-1]["header_fields_raw"] = page_output.header_fields_raw
                rows.extend(page_rows)
                page_output.reconstructed_table = [self._to_row_output(r) for r in page_rows]
                summary.pages.append(page_output)
                summary.pages_seen += 1
            summary.rows_emitted = len(rows)
            summary.structural_row_metrics = self._summarize_document_row_metrics(summary.pages)
            page_warnings = [w for p in summary.pages for w in p.warnings]
            for doc_warning in ("excessive_stitching_detected", "large_row_count_drop", "layout_low_confidence", "parser_conflict_detected"):
                if doc_warning in page_warnings and doc_warning not in summary.warnings:
                    summary.warnings.append(doc_warning)
            return rows, summary
        except Exception as exc:
            summary.errors.append(f"document_error:{type(exc).__name__}:{exc}")
            self.logger.error("document parse failed", extra={"structured": {"source_file": str(pdf_path), "error": str(exc)}})
            if not self.config.continue_on_error:
                raise
            return rows, summary
        finally:
            doc.close()

    def _parse_page(self, pdf_path: Path, page_ctx: PageContext, summary: DocumentSummary, page_output: PageOutput) -> list[RawRowRecord]:
        parser_results = []
        for parser in self.parsers:
            try:
                result = parser.parse_page(pdf_path, page_ctx)
                parser_results.append(result)
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

        selected, decision = self.fuser.choose(page_ctx.page_number, parser_results)
        summary.fusion_decisions.append(decision)
        page_output.parser_decision = decision.model_dump()
        summary.parser_usage[decision.selected_parser] = summary.parser_usage.get(decision.selected_parser, 0) + 1

        if decision.disagreement:
            page_output.warnings.append("parser_disagreement")

        normalized: list[RawRowRecord] = []
        candidate_lines = max((len(r.rows) for r in parser_results), default=len(selected.rows))
        reconstructed_rows, structure_warnings, boundaries = apply_structure_assisted_reconstruction(selected.rows, parser_results)
        for warning in structure_warnings:
            if warning not in page_output.warnings:
                page_output.warnings.append(warning)
        page_output.layout_model["column_boundaries"] = boundaries

        for row in reconstructed_rows:
            row = weak_map_columns(row)
            row = validate_row(row)
            if decision.disagreement and "parser_disagreement" not in row.warnings:
                row.warnings.append("parser_disagreement")
            normalized.append(row)

        normalized, lane_metrics = apply_page_lane_inference(normalized)

        boundary_rows, boundary_warnings, boundary_metrics = apply_row_boundary_engine(normalized, parser_results)
        normalized = stitch_multiline_rows(boundary_rows)
        boundary_metrics.update(continuation_metrics(normalized))
        boundary_metrics.update(lane_metrics)

        for warning in boundary_warnings:
            if warning not in page_output.warnings:
                page_output.warnings.append(warning)
        page_output.layout_model["row_boundary_metrics"] = boundary_metrics

        if candidate_lines:
            drop_ratio = max(0.0, 1 - (len(normalized) / candidate_lines))
            if drop_ratio > 0.35 and "row_count_sanity_check_failed" not in page_output.warnings:
                page_output.warnings.append("row_count_sanity_check_failed")
            if drop_ratio > 0.5 and "large_row_count_drop" not in page_output.warnings:
                page_output.warnings.append("large_row_count_drop")

        if any("excessive_row_merge_detected" in r.warnings for r in normalized) and "excessive_stitching_detected" not in page_output.warnings:
            page_output.warnings.append("excessive_stitching_detected")
        if page_output.layout_model.get("layout_confidence", 1.0) < 0.45 and "layout_low_confidence" not in page_output.warnings:
            page_output.warnings.append("layout_low_confidence")
        if any("ambiguous_row_boundary" in r.warnings for r in normalized) and "parser_conflict_detected" not in page_output.warnings:
            page_output.warnings.append("parser_conflict_detected")

        return self._decontaminate_page_rows(normalized)

    def _decontaminate_page_rows(self, rows: list[RawRowRecord]) -> list[RawRowRecord]:
        if not rows:
            return rows
        cleaned: list[RawRowRecord] = []
        for row in rows:
            atomic = row.metadata.get("atomic_line") if isinstance(row.metadata, dict) else None
            starts_anchor = isinstance(atomic, dict) and atomic.get("starts_with_item_anchor") is True
            structural_state = row.metadata.get("row_structure_classification") if isinstance(row.metadata, dict) else None
            if "header_row" in row.warnings:
                for flag in ("probable_header_contamination", "header_contamination_detected", "probable_header_leakage"):
                    if flag not in row.warnings:
                        row.warnings.append(flag)
            if "footer_row" in row.warnings:
                for flag in ("probable_footer_contamination", "footer_contamination_detected"):
                    if flag not in row.warnings:
                        row.warnings.append(flag)

            suppress_from_table = (
                structural_state in {"table_header_row", "non_bom_structural_row"}
                or (("probable_header_leakage" in row.warnings or "header_row" in row.warnings)
                and not starts_anchor
                )
            )
            if suppress_from_table:
                continue
            cleaned.append(row)
        return cleaned

    def _build_page_layout(self, page: fitz.Page, page_ctx: PageContext) -> dict:
        words = page.get_text("words") or []
        compact_words = [(w[0], w[1], w[2], w[3], normalize_space(w[4])) for w in words if normalize_space(w[4])]
        layout = infer_page_layout(page.rect.height, compact_words)
        return {
            "zone_header_cutoff": layout.zones.header_cutoff,
            "zone_footer_cutoff": layout.zones.footer_cutoff,
            "header_lines": layout.header_lines,
            "table_lines": layout.table_lines,
            "footer_lines": layout.footer_lines,
            "background_noise_lines": layout.background_noise_lines,
            "layout_warnings": layout.warnings,
            "layout_confidence": layout.confidence,
            "page_width": page_ctx.page_width,
            "page_height": page_ctx.page_height,
            "page_words": compact_words,
        }

    def _to_row_output(self, row: RawRowRecord) -> RowOutput:
        return RowOutput(
            source_file=row.source_file,
            page_number=row.page_number,
            row_index=row.row_index_on_page,
            raw_fragments=row.metadata.get("raw_fragments", list(row.extracted_columns)),
            reconstructed_text=row.raw_text,
            extracted_columns=row.extracted_columns,
            parser_sources=row.metadata.get("parser_sources", [row.parser_name]),
            confidence=row.parser_confidence,
            confidence_band=row.metadata.get("operational_confidence_band"),
            structural_state=row.metadata.get("row_structure_classification"),
            warnings=row.warnings,
        )

    @staticmethod
    def _merge_parser_usage(summaries: list[DocumentSummary]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for summary in summaries:
            for key, value in summary.parser_usage.items():
                merged[key] = merged.get(key, 0) + value
        return merged

    @staticmethod
    def _merge_counter_maps(summaries: list[DocumentSummary], field_name: str) -> dict[str, int]:
        merged: dict[str, int] = {}
        for summary in summaries:
            counter = getattr(summary, field_name, {}) or {}
            for key, value in counter.items():
                merged[key] = merged.get(key, 0) + int(value)
        return merged

    @staticmethod
    def _merge_metric_maps(summaries: list[DocumentSummary]) -> dict[str, float | int]:
        merged: dict[str, float | int] = {}
        for summary in summaries:
            for key, value in (summary.structural_row_metrics or {}).items():
                merged[key] = merged.get(key, 0) + value
        return merged

    @staticmethod
    def _classify_page_state(page_output: PageOutput, rows: list[RawRowRecord], previous_page_state: str | None) -> str:
        header_metrics = page_output.layout_model.get("header_extraction_metrics", {})
        fields_detected = int(header_metrics.get("header_fields_detected", 0) or 0)
        boundary_conflicts = int(header_metrics.get("header_boundary_conflicts", 0) or 0)
        leaked_bom_like_header_raw = any(normalize_space(line).split(" ", 1)[0].isdigit() for line in page_output.header_raw_lines if normalize_space(line))
        row_states = [r.metadata.get("row_structure_classification") for r in rows if isinstance(r.metadata, dict)]
        has_bom_rows = any(
            state in {"clean_anchor_row", "anchor_row_with_expandable_continuation", "continuation_fragment_row", "ambiguous_row"}
            for state in row_states
        )
        has_footer_only_rows = bool(rows) and all(state == "non_bom_structural_row" for state in row_states if state)

        if fields_detected == 0 and has_bom_rows and leaked_bom_like_header_raw:
            if "header_raw_table_leakage" not in page_output.warnings:
                page_output.warnings.append("header_raw_table_leakage")
            return "continuation_page_without_header"
        if boundary_conflicts > 0:
            return "page_with_header_boundary_uncertainty"
        if fields_detected > 0:
            return "page_with_primary_header"
        if has_footer_only_rows or (page_output.footer_fields_raw and not has_bom_rows):
            return "page_with_footer_only"
        if has_bom_rows and previous_page_state in {"page_with_primary_header", "continuation_page_without_header"}:
            return "continuation_page_without_header"
        if has_bom_rows:
            return "continuation_page_without_header"
        return "page_with_footer_only"

    @staticmethod
    def _apply_page_header_policy(page_output: PageOutput) -> None:
        if page_output.page_state != "continuation_page_without_header":
            return
        page_output.header_code = None
        page_output.header_revision = None
        page_output.header_type = None
        page_output.header_description = None
        page_output.header_fields_raw = []
        page_output.header_raw_lines = []

    @staticmethod
    def _summarize_document_row_metrics(pages: list[PageOutput]) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "clean_anchor_row_count": 0,
            "continuation_row_count": 0,
            "ambiguous_row_count": 0,
            "continuation_fragment_count": 0,
            "table_header_row_count": 0,
            "non_bom_structural_row_count": 0,
            "high_confidence_row_count": 0,
            "medium_confidence_row_count": 0,
            "low_confidence_row_count": 0,
        }
        for page in pages:
            row_metrics = page.layout_model.get("row_boundary_metrics", {})
            for key in metrics:
                metrics[key] += int(row_metrics.get(key, 0) or 0)
        return metrics
