from pathlib import Path

from bom_extractor.config import ExtractionConfig
from bom_extractor.pipeline import ExtractionPipeline


PDFS = [
    Path('/mnt/data/E0080082 01-06_BOM.pdf'),
    Path('/mnt/data/E0047678 01-01_BOM.pdf'),
    Path('/mnt/data/E0181296 01-06_BOM.pdf'),
]


def test_pipeline_smoke_real_pdfs(tmp_path):
    available = [p for p in PDFS if p.exists()]
    if not available:
        return

    config = ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False)
    pipeline = ExtractionPipeline(config)
    rows = pipeline.parse_input(available[0])

    assert rows, 'Expected at least one extracted row'
    assert any(r.raw_text for r in rows)
    assert all(r.source_file for r in rows)
    assert all(r.page_number >= 1 for r in rows)
