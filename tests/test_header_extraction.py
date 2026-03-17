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
