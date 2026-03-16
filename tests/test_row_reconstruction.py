from bom_extractor.models import RawRowRecord
from bom_extractor.normalization import stitch_multiline_rows


def _row(
    idx: int,
    text: str,
    cols: list[str],
    warnings: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
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


def test_stitch_marks_ambiguous_alignment_when_bboxes_do_not_overlap():
    rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], bbox=(10, 100, 120, 108)),
        _row(2, "WRAPPED", ["WRAPPED"], ["continuation_candidate"], bbox=(300, 112, 380, 120)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 1
    assert "ambiguous_alignment" in out[0].warnings
    assert "possible_fragmentation" in out[0].warnings


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
