from bom_extractor.models import RawRowRecord
from bom_extractor.normalizer import weak_map_columns


def _row(text: str, cols: list[str]) -> RawRowRecord:
    return RawRowRecord(
        source_file="a.pdf",
        source_file_hash="h",
        document_id="d",
        page_number=1,
        row_index_on_page=1,
        raw_text=text,
        extracted_columns=cols,
        parser_confidence=0.7,
        parser_name="x",
        warnings=[],
    )


def test_weak_map_columns_extracts_kgm_quantity_005():
    row = _row("0010 TYPE E0181296 01 KGM 0.05", ["0010", "TYPE", "E0181296", "01", "KGM", "0.05"])
    out = weak_map_columns(row)
    assert out.uom == "KGM"
    assert out.quantity_raw == "0.05"


def test_weak_map_columns_extracts_kgm_quantity_0025():
    row = _row("0010 TYPE E0181296 01 KGM 0.025", ["0010", "TYPE", "E0181296", "01", "KGM", "0.025"])
    out = weak_map_columns(row)
    assert out.uom == "KGM"
    assert out.quantity_raw == "0.025"


def test_weak_map_columns_preserves_literal_null_anchor_row():
    row = _row(
        "null Disegno E0181296 01.DRW 14 QUIRIS Laser Unit",
        ["null", "Disegno", "E0181296 01.DRW", "14", "QUIRIS Laser Unit"],
    )
    out = weak_map_columns(row)
    assert out.item == "null"
    assert out.type_raw == "Disegno"
    assert out.code == "E0181296 01.DRW"
    assert out.revision == "14"
