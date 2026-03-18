from pathlib import Path

from bom_extractor.config import ExtractionConfig
from bom_extractor.pipeline import ExtractionPipeline


def test_pipeline_smoke_repo_sample(tmp_path):
    sample = Path("E0181296 01-06_BOM.pdf")
    if not sample.exists():
        return

    config = ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False, max_pages=2)
    pipeline = ExtractionPipeline(config)
    rows, summary = pipeline.parse_document(sample)

    assert rows, "Expected at least one extracted row"
    assert all(r.raw_text for r in rows)
    assert all(r.source_file for r in rows)
    assert all(r.page_number >= 1 for r in rows)
    assert all(r.parser_name for r in rows)
    assert any(r.extracted_columns for r in rows)

    # Hard item-anchor boundaries should remain one logical row.
    for row in rows:
        assert not (row.item and row.raw_text.count(row.item) > 1 and "multi_item_row_detected" not in row.warnings)

    # Header extraction is targeted to only the expected structured fields.
    assert summary.pages
    header_page = summary.pages[0]
    assert hasattr(header_page, "header_code")
    assert hasattr(header_page, "header_revision")
    assert hasattr(header_page, "header_type")
    assert hasattr(header_page, "header_description")

    metrics = header_page.layout_model.get("row_boundary_metrics", {})
    for key in (
        "isolated_continuation_count",
        "attached_continuation_count",
        "uncertain_continuation_count",
        "continuation_attachment_rate",
        "anchor_field_duplication_events",
        "lane_count",
        "lane_confidence_score",
        "field_assignment_uncertain_count",
        "anchor_lane_conflict_count",
        "rows_with_clean_anchor_alignment",
        "expandable_field_attachment_count",
        "clean_anchor_row_count",
        "continuation_row_count",
        "ambiguous_row_count",
        "continuation_fragment_count",
        "table_header_row_count",
        "non_bom_structural_row_count",
        "high_confidence_row_count",
        "medium_confidence_row_count",
        "low_confidence_row_count",
        "warning_density",
        "noisy_warning_suppression_count",
    ):
        assert key in metrics

    assert metrics["lane_count"] >= 1
    assert summary.page_state_distribution
    assert header_page.page_state is not None
