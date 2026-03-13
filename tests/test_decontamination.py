from pathlib import Path

from bom_extractor.config import ExtractionConfig
from bom_extractor.models import RawRowRecord
from bom_extractor.pipeline import ExtractionPipeline


def test_decontaminate_prefers_clean_rows(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=1, raw_text="header", extracted_columns=["header"], parser_name="x", warnings=["header_row"]),
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=2, raw_text="0010 MAT 123456", extracted_columns=["0010", "MAT", "123456"], parser_name="x"),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert len(out) == 1
    assert out[0].raw_text.startswith("0010")
