from pathlib import Path

from bom_extractor.config import ExtractionConfig
from bom_extractor.pipeline import ExtractionPipeline


def test_pipeline_smoke_repo_sample(tmp_path):
    sample = Path("E0181296 01-06_BOM.pdf")
    if not sample.exists():
        return

    config = ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False, max_pages=2)
    pipeline = ExtractionPipeline(config)
    rows = pipeline.parse_input(sample)

    assert rows, "Expected at least one extracted row"
    assert all(r.raw_text for r in rows)
    assert all(r.source_file for r in rows)
    assert all(r.page_number >= 1 for r in rows)
    assert all(r.parser_name for r in rows)
    assert any(r.extracted_columns for r in rows)
