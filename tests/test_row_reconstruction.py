from bom_extractor.models import RawRowRecord
from bom_extractor.normalization import stitch_multiline_rows


def _row(
    idx: int,
    text: str,
    cols: list[str],
    warnings: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    **fields,
) -> RawRowRecord:
    return RawRowRecord(
        source_file="a.pdf",
        source_file_hash="h",
        document_id="d",
        page_number=1,
        row_index_on_page=idx,
        raw_text=text,
        extracted_columns=cols,
        parser_confidence=0.7,
        parser_name="x",
        warnings=warnings or [],
        bbox_row=bbox,
        **fields,
    )


def test_stitch_preserves_fragment_evidence():
    rows = [
        _row(1, "0010 MAT BODY", ["0010", "MAT", "BODY"], bbox=(10, 100, 200, 108)),
        _row(2, "CONTINUATION DETAILS", ["CONTINUATION", "DETAILS"], ["continuation_candidate"], bbox=(12, 112, 210, 120)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 1
    assert "CONTINUATION DETAILS" in out[0].raw_text
    assert "row_stitched" in out[0].warnings
    assert out[0].metadata.get("stitched_fragments")


def test_stitch_does_not_merge_when_alignment_is_ambiguous():
    rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], bbox=(10, 100, 120, 108)),
        _row(2, "WRAPPED", ["WRAPPED"], ["continuation_candidate"], bbox=(300, 112, 380, 120)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2
    assert "ambiguous_alignment" in out[0].warnings


def test_does_not_stitch_when_new_item_token_appears():
    rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], bbox=(10, 100, 120, 108)),
        _row(2, "0020 NEXT", ["0020", "NEXT"], ["continuation_candidate"], bbox=(10, 111, 120, 119)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2


def test_does_not_stitch_when_vertical_gap_too_large():
    rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], bbox=(10, 100, 120, 108)),
        _row(2, "DETAILS", ["DETAILS"], ["continuation_candidate"], bbox=(10, 140, 120, 148)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2
    assert "merge_blocked_vertical_gap" in out[0].warnings


def test_limits_excessive_fragment_stitching():
    rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], bbox=(10, 100, 120, 108)),
        _row(2, "A", ["A"], ["continuation_candidate"], bbox=(10, 110, 120, 118)),
        _row(3, "B", ["B"], ["continuation_candidate"], bbox=(10, 120, 120, 128)),
        _row(4, "C", ["C"], ["continuation_candidate"], bbox=(10, 130, 120, 138)),
        _row(5, "D", ["D"], ["continuation_candidate"], bbox=(10, 140, 120, 148)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2
    assert "excessive_row_merge_detected" in out[0].warnings


def test_continuation_does_not_overwrite_anchor_fields_when_locked():
    rows = [
        _row(
            1,
            "0010 TYPE E0181296 01 NR 2",
            ["0010", "TYPE", "E0181296", "01", "NR", "2"],
            bbox=(10, 100, 240, 108),
            item="0010",
            code="E0181296",
            revision="01",
            uom="NR",
            quantity_raw="2",
            description="BASE DESC",
        ),
        _row(
            2,
            "WRONG E9999999 03 KG 9 extra notes",
            ["WRONG", "E9999999", "03", "KG", "9", "extra", "notes"],
            ["continuation_candidate"],
            bbox=(20, 111, 250, 119),
            code="E9999999",
            revision="03",
            uom="KG",
            quantity_raw="9",
            notes="extra notes",
        ),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 1
    merged = out[0]
    assert merged.item == "0010"
    assert merged.code == "E0181296"
    assert merged.revision == "01"
    assert merged.uom == "NR"
    assert merged.quantity_raw == "2"
    assert "anchor_field_duplication_suspected" in merged.warnings
    assert "row_anchor_fields_locked" in merged.warnings
    assert "continuation_to_expandable_field" in merged.warnings


def test_continuation_targets_company_or_notes_by_lexical_shape():
    rows = [
        _row(1, "0010 TYPE E0181296 01 NR 2", ["0010"], bbox=(10, 100, 220, 108), item="0010", code="E0181296", revision="01"),
        _row(2, "supplier ACME SRL", ["supplier", "ACME", "SRL"], ["continuation_candidate"], bbox=(100, 111, 250, 119)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 1
    assert out[0].company_name
    assert "ACME" in out[0].company_name
