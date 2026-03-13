from bom_extractor.fusion import PageResultFuser
from bom_extractor.models import ParserPageResult, RawRowRecord


def make_row(text: str, cols: list[str], parser: str = "p") -> RawRowRecord:
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
