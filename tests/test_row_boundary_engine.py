from bom_extractor.models import ParserPageResult, RawRowRecord
from bom_extractor.normalization.row_boundary_engine import apply_row_boundary_engine


def _row(idx: int, text: str, cols: list[str], bbox: tuple[float, float, float, float], parser: str = "pymupdf_words") -> RawRowRecord:
    return RawRowRecord(
        source_file="a.pdf",
        source_file_hash="h",
        document_id="d",
        page_number=1,
        row_index_on_page=idx,
        raw_text=text,
        extracted_columns=cols,
        parser_confidence=0.7,
        parser_name=parser,
        bbox_row=bbox,
    )


def test_item_anchor_lines_are_hard_row_boundaries():
    rows = [
        _row(1, "0015 VALVE", ["0015", "VALVE"], (10, 100, 200, 108)),
        _row(2, "0020 MOTOR", ["0020", "MOTOR"], (10, 111, 200, 119)),
        _row(3, "0025 PUMP", ["0025", "PUMP"], (10, 123, 200, 131)),
    ]
    parser_results = [ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=rows)]

    out, warnings, metrics = apply_row_boundary_engine(rows, parser_results)

    assert len(out) == 3
    assert metrics["candidate_item_anchor_count"] == 3
    assert "item_column_uncertain" not in warnings


def test_literal_null_item_is_treated_as_anchor_boundary():
    rows = [
        _row(1, "0015 VALVE", ["0015", "VALVE"], (10, 100, 200, 108)),
        _row(2, "null Disegno E0181296 01.DRW 14", ["null", "Disegno", "E0181296", "01.DRW", "14"], (10, 111, 240, 119)),
    ]
    parser_results = [ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=rows)]

    out, _, metrics = apply_row_boundary_engine(rows, parser_results)

    assert len(out) == 2
    assert metrics["candidate_item_anchor_count"] == 2
    assert out[1].metadata["atomic_line"]["starts_with_item_anchor"] is True


def test_row_loss_warning_when_anchor_count_far_exceeds_rows():
    selected = [_row(1, "DESC ONLY", ["DESC", "ONLY"], (90, 180, 210, 188))]
    secondary_rows = [
        _row(1, "0010 A", ["0010", "A"], (10, 100, 120, 108), parser="pdfplumber_table"),
        _row(2, "0020 B", ["0020", "B"], (10, 112, 120, 120), parser="pdfplumber_table"),
        _row(3, "0030 C", ["0030", "C"], (10, 124, 120, 132), parser="pdfplumber_table"),
    ]
    parser_results = [
        ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=selected),
        ParserPageResult(parser_name="pdfplumber_table", page_number=1, rows=secondary_rows),
    ]

    out, warnings, _ = apply_row_boundary_engine(selected, parser_results)

    assert len(out) == 1
    assert "boundary_disagreement" in warnings


def test_continuation_fragment_attaches_and_marks_warning():
    rows = [
        _row(1, "0010 BODY WITH CODE 123456", ["0010", "BODY", "123456"], (10, 100, 200, 108)),
        _row(2, "10_35", ["10_35"], (88, 110, 130, 118)),
    ]
    parser_results = [ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=rows)]

    out, _, _ = apply_row_boundary_engine(rows, parser_results)

    assert len(out) == 1
    assert "10_35" in out[0].raw_text
    assert "continuation_attached" in out[0].warnings


def test_uncertain_continuation_is_not_auto_attached():
    rows = [
        _row(1, "0010 BODY WITH CODE 123456", ["0010", "BODY", "123456"], (10, 100, 200, 108)),
        _row(2, "trade continuation", ["trade", "continuation"], (300, 128, 380, 136)),
    ]
    parser_results = [ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=rows)]

    out, _, _ = apply_row_boundary_engine(rows, parser_results)

    assert len(out) == 2
    assert "continuation_attachment_uncertain" in out[1].warnings


def test_parser_supported_attachment_warning_is_emitted():
    primary_rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], (10, 100, 200, 108), parser="pymupdf_words"),
        _row(2, "supplier ACME LTD", ["supplier", "ACME", "LTD"], (90, 110, 230, 118), parser="pymupdf_words"),
    ]
    secondary_rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], (10, 100, 200, 108), parser="pdfplumber_table"),
        _row(2, "supplier ACME LTD", ["supplier", "ACME", "LTD"], (90, 111, 230, 119), parser="pdfplumber_table"),
    ]
    parser_results = [
        ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=primary_rows),
        ParserPageResult(parser_name="pdfplumber_table", page_number=1, rows=secondary_rows),
    ]

    out, _, _ = apply_row_boundary_engine(primary_rows, parser_results)

    assert len(out) == 1
    assert "parser_supported_attachment" in out[0].warnings


def test_table_header_anchor_is_not_treated_as_data_row():
    rows = [
        _row(1, "ITEM CODE QTY DESCRIPTION", ["ITEM", "CODE", "QTY", "DESCRIPTION"], (10, 90, 260, 98)),
        _row(2, "0010 VALVE", ["0010", "VALVE"], (10, 102, 180, 110)),
    ]
    parser_results = [ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=rows)]

    out, _, _ = apply_row_boundary_engine(rows, parser_results)

    assert "probable_header_leakage" in out[0].warnings
    assert out[1].raw_text.startswith("0010")


def test_multi_item_warning_ignores_reference_fragments():
    rows = [
        _row(1, "0010 BRACKET REF 10_35 5", ["0010", "BRACKET", "10_35", "5"], (10, 100, 220, 108)),
    ]
    parser_results = [ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=rows)]

    out, _, _ = apply_row_boundary_engine(rows, parser_results)

    assert len(out) == 1
    assert "multi_item_row_detected" not in out[0].warnings
