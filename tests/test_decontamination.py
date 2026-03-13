from bom_extractor.config import ExtractionConfig
from bom_extractor.models import RawRowRecord
from bom_extractor.pipeline import ExtractionPipeline


def test_decontaminate_marks_contamination_but_preserves_rows(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=1, raw_text="BILL OF MATERIAL", extracted_columns=["BILL OF MATERIAL"], parser_name="x", warnings=["header_row"]),
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=2, raw_text="0010 MAT 123456", extracted_columns=["0010", "MAT", "123456"], parser_name="x"),
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=3, raw_text="Rev. date 2024", extracted_columns=["Rev.", "date", "2024"], parser_name="x", warnings=["footer_row"]),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert len(out) == 3
    assert "probable_header_contamination" in out[0].warnings
    assert "probable_footer_contamination" in out[2].warnings
