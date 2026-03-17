from bom_extractor.normalization.header_extraction import extract_targeted_header_fields


def test_extract_targeted_header_fields_bilingual_patterns():
    lines = [
        "BILL OF MATERIAL",
        "Codice: E0181296",
        "Rev: 01",
        "Tipo: ASSY",
        "Descrizione: SUPPORT BRACKET",
    ]
    out = extract_targeted_header_fields(lines)
    assert out["header_code"] == "E0181296"
    assert out["header_revision"] == "01"
    assert out["header_type"] == "ASSY"
    assert out["header_description"] == "SUPPORT BRACKET"
    assert out["header_raw_lines"]


def test_geometry_driven_header_extraction_stops_before_table_header():
    words = [
        (20.0, 20.0, 80.0, 28.0, "Codice"),
        (95.0, 20.0, 170.0, 28.0, "E0181296"),
        (20.0, 34.0, 95.0, 42.0, "Revisione"),
        (100.0, 34.0, 120.0, 42.0, "01"),
        (20.0, 48.0, 60.0, 56.0, "Tipo"),
        (95.0, 48.0, 150.0, 56.0, "ASSY"),
        (20.0, 62.0, 120.0, 70.0, "Descrizione"),
        (130.0, 62.0, 260.0, 70.0, "SUPPORT BRACKET"),
        (20.0, 98.0, 45.0, 106.0, "Riga"),
        (50.0, 98.0, 80.0, 106.0, "Item"),
        (86.0, 98.0, 130.0, 106.0, "Codice"),
        (135.0, 98.0, 170.0, 106.0, "Code"),
        (175.0, 98.0, 230.0, 106.0, "Descrizione"),
        (235.0, 98.0, 300.0, 106.0, "Description"),
    ]
    out = extract_targeted_header_fields([], words=words, page_height=800.0)
    assert out["header_code"] == "E0181296"
    assert out["header_revision"] == "01"
    assert out["header_type"] == "ASSY"
    assert out["header_description"] == "SUPPORT BRACKET"
    assert out["header_bbox"] is not None
    assert not any("Riga Item" in line for line in out["header_raw_lines"])
    assert out["header_fields_detected"] == 4


def test_header_boundary_uncertain_warning_when_no_anchors():
    words = [
        (20.0, 20.0, 80.0, 28.0, "Codice"),
        (90.0, 20.0, 150.0, 28.0, "E0181296"),
    ]
    out = extract_targeted_header_fields([], words=words, page_height=800.0)
    assert "header_boundary_uncertain" in out["warnings"]
