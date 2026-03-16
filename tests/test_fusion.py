from bom_extractor.fusion import PageResultFuser
from bom_extractor.models import ParserPageResult, RawRowRecord
from bom_extractor.normalization.table_structure import apply_structure_assisted_reconstruction


def make_row(text: str, cols: list[str], parser: str = "p", metadata=None, bbox=None) -> RawRowRecord:
    return RawRowRecord(
        source_file="a.pdf",
        source_file_hash="h",
        document_id="d",
        page_number=1,
        row_index_on_page=1,
        raw_text=text,
        extracted_columns=cols,
        parser_confidence=0.5,
        parser_name=parser,
        metadata=metadata or {},
        bbox_row=bbox,
    )


def test_fuser_prefers_data_like_rows():
    bad = ParserPageResult(
        parser_name="bad",
        page_number=1,
        confidence=0.9,
        rows=[make_row("Riga Item Tipo Code", ["Riga", "Item", "Tipo"], "bad")],
    )
    good = ParserPageResult(
        parser_name="good",
        page_number=1,
        confidence=0.6,
        rows=[make_row("0010 MAT 123456 REV BRACKET NR 2", ["0010", "MAT", "123456", "REV", "BRACKET", "NR", "2"], "good")],
    )
    selected, decision = PageResultFuser().choose(1, [bad, good])
    assert selected.parser_name == "good"
    assert decision.selected_parser == "good"


def test_structure_assisted_reconstruction_exposes_parser_sources():
    selected_rows = [
        make_row(
            "0010 BOLT M10", ["0010", "BOLT", "M10"], "pymupdf_words", metadata={
                "word_boxes": [
                    {"x0": 10, "text": "0010"},
                    {"x0": 80, "text": "BOLT"},
                    {"x0": 170, "text": "M10"},
                ]
            }
        )
    ]
    parser_results = [
        ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=selected_rows, metadata={"column_x_hints": [10, 80, 170]}),
        ParserPageResult(parser_name="pdfplumber_table", page_number=1, rows=[], metadata={"column_count_hint": 3}),
    ]

    reconstructed, warnings, boundaries = apply_structure_assisted_reconstruction(selected_rows, parser_results)

    assert not warnings
    assert boundaries
    assert reconstructed[0].metadata["parser_sources"] == ["pdfplumber_table", "pymupdf_words"]


def test_structure_marks_ambiguous_boundaries_on_parser_disagreement():
    selected_rows = [
        make_row("0010 BOLT", ["0010", "BOLT"], "pymupdf_words", bbox=(10, 120, 150, 130)),
    ]
    parser_results = [
        ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=selected_rows, metadata={"column_x_hints": [10, 80]}),
        ParserPageResult(
            parser_name="pdfplumber_table",
            page_number=1,
            rows=[make_row("0010 BOLT", ["0010", "BOLT"], "pdfplumber_table", bbox=(12, 180, 150, 190))],
            metadata={"column_count_hint": 2},
        ),
    ]

    reconstructed, warnings, _ = apply_structure_assisted_reconstruction(selected_rows, parser_results)
    assert "ambiguous_row_boundary" in reconstructed[0].warnings
    assert "parser_conflict_detected" in warnings


def test_structure_emits_row_count_sanity_warning_on_large_drop():
    selected_rows = [make_row("0010", ["0010"], "pymupdf_words")]
    parser_results = [
        ParserPageResult(parser_name="pymupdf_words", page_number=1, rows=selected_rows),
        ParserPageResult(
            parser_name="pdfplumber_table",
            page_number=1,
            rows=[make_row(str(i), [str(i)], "pdfplumber_table") for i in range(10)],
        ),
    ]

    _, warnings, _ = apply_structure_assisted_reconstruction(selected_rows, parser_results)
    assert "row_count_sanity_check_failed" in warnings
    assert "large_row_count_drop" in warnings
