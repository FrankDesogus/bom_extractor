from bom_extractor.config import ExtractionConfig
from bom_extractor.models import PageOutput, RawRowRecord
from bom_extractor.pipeline import ExtractionPipeline


def test_decontaminate_marks_contamination_and_suppresses_header_rows(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=1, raw_text="BILL OF MATERIAL", extracted_columns=["BILL OF MATERIAL"], parser_name="x", warnings=["header_row"]),
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=2, raw_text="0010 MAT 123456", extracted_columns=["0010", "MAT", "123456"], parser_name="x", item="0010"),
        RawRowRecord(source_file="a", source_file_hash="h", document_id="d", page_number=1, row_index_on_page=3, raw_text="Rev. date 2024", extracted_columns=["Rev.", "date", "2024"], parser_name="x", warnings=["footer_row"]),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert len(out) == 1
    assert out[0].row_index_on_page == 2
    assert "probable_footer_contamination" in rows[2].warnings


def test_decontaminate_suppresses_structural_non_bom_rows(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=1,
            raw_text="ITEM CODE QTY",
            extracted_columns=["ITEM", "CODE", "QTY"],
            parser_name="x",
            warnings=["table_header_row"],
            metadata={"row_structure_classification": "table_header_row", "atomic_line": {"starts_with_item_anchor": False}},
        ),
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=2,
            raw_text="0010 MAT 123456",
            extracted_columns=["0010", "MAT", "123456"],
            parser_name="x",
            item="0010",
            metadata={"row_structure_classification": "clean_anchor_row", "atomic_line": {"starts_with_item_anchor": True}},
        ),
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=3,
            raw_text="Legal footer",
            extracted_columns=["Legal", "footer"],
            parser_name="x",
            metadata={"row_structure_classification": "non_bom_structural_row"},
        ),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert len(out) == 1
    assert out[0].row_index_on_page == 2


def test_decontaminate_keeps_footer_adjacent_valid_anchor_row(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=10,
            raw_text="0165 MAT E0181296 01 NR 2",
            extracted_columns=["0165", "MAT", "E0181296", "01", "NR", "2"],
            parser_name="x",
            item="0165",
            code="E0181296",
            revision="01",
            warnings=["footer_row"],
            metadata={"row_structure_classification": "non_bom_structural_row", "atomic_line": {"is_footer_like": True}},
        ),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert len(out) == 1
    assert out[0].item == "0165"


def test_decontaminate_keeps_footer_adjacent_valid_anchor_row_with_literal_null_item(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=11,
            raw_text="null Disegno E0181296 01.DRW 14 QUIRIS Laser Unit",
            extracted_columns=["null", "Disegno", "E0181296 01.DRW", "14", "QUIRIS Laser Unit"],
            parser_name="x",
            item="null",
            type_raw="Disegno",
            code="E0181296 01.DRW",
            revision="14",
            description="QUIRIS Laser Unit",
            warnings=["footer_row"],
            metadata={"row_structure_classification": "non_bom_structural_row", "atomic_line": {"is_footer_like": True}},
        ),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert len(out) == 1
    assert out[0].item == "null"
    assert out[0].type_raw == "Disegno"
    assert out[0].code == "E0181296 01.DRW"
    assert out[0].revision == "14"


def test_decontaminate_drops_footer_legal_line_even_if_row_like(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=12,
            raw_text="THIS DOCUMENT CONTAINS PROPRIETARY INFORMATION E0181296 06",
            extracted_columns=["0165", "THIS DOCUMENT CONTAINS PROPRIETARY INFORMATION", "E0181296", "06"],
            parser_name="x",
            item="0165",
            code="E0181296",
            revision="06",
            warnings=["footer_row"],
            metadata={"row_structure_classification": "non_bom_structural_row", "atomic_line": {"is_footer_like": True}},
        ),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert out == []


def test_decontaminate_drops_stage_metadata_line_even_if_row_like(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=13,
            raw_text="Stato/Stage: RILASCIATO / RELEASED Nome/Code: E0181296 01 Rev: 06",
            extracted_columns=["0165", "Stato/Stage:", "RILASCIATO", "E0181296", "06"],
            parser_name="x",
            item="0165",
            code="E0181296",
            revision="06",
            warnings=["footer_row"],
            metadata={"row_structure_classification": "non_bom_structural_row", "atomic_line": {"is_footer_like": True}},
        ),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert out == []


def test_decontaminate_drops_rows_with_missing_item(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=1,
            row_index_on_page=14,
            raw_text="Disegno E0181296 01.DRW 14 QUIRIS Laser Unit",
            extracted_columns=["Disegno", "E0181296 01.DRW", "14", "QUIRIS Laser Unit"],
            parser_name="x",
            item=None,
            code="E0181296 01.DRW",
            revision="14",
            metadata={"row_structure_classification": "clean_anchor_row", "atomic_line": {"starts_with_item_anchor": False}},
        ),
    ]
    out = pipe._decontaminate_page_rows(rows)
    assert out == []


def test_continuation_page_header_policy_nulls_structured_header_fields(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    page = PageOutput(
        page_number=2,
        page_state="continuation_page_without_header",
        header_code="E0181296",
        header_revision="01",
        header_type="ASSY",
        header_description="SHOULD BE CLEARED",
        header_fields_raw=["0010 leaked row"],
        header_raw_lines=["0010 leaked row"],
    )
    pipe._apply_page_header_policy(page)
    assert page.header_code is None
    assert page.header_revision is None
    assert page.header_type is None
    assert page.header_description is None
    assert page.header_fields_raw == []
    assert page.header_raw_lines == []


def test_page_state_classifier_detects_continuation_page_without_header(tmp_path):
    pipe = ExtractionPipeline(ExtractionConfig(output_dir=tmp_path, write_csv=False, write_parquet=False))
    rows = [
        RawRowRecord(
            source_file="a",
            source_file_hash="h",
            document_id="d",
            page_number=2,
            row_index_on_page=1,
            raw_text="0010 MAT E0181296 01 NR 2",
            extracted_columns=["0010", "MAT", "E0181296", "01", "NR", "2"],
            parser_name="x",
            metadata={"row_structure_classification": "clean_anchor_row"},
        )
    ]
    page = PageOutput(
        page_number=2,
        layout_model={"header_extraction_metrics": {"header_fields_detected": 0, "header_boundary_conflicts": 0}},
        footer_fields_raw=[],
    )
    state = pipe._classify_page_state(page, rows, "page_with_primary_header")
    assert state == "continuation_page_without_header"
