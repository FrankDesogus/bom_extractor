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
