from bom_extractor.models import RawRowRecord
from bom_extractor.normalization import stitch_multiline_rows


def _row(idx: int, text: str, cols: list[str], warnings: list[str] | None = None) -> RawRowRecord:
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
    )


def test_stitch_preserves_fragment_evidence():
    rows = [
        _row(1, "0010 MAT 123456 BODY", ["0010", "MAT", "123456", "BODY"]),
        _row(2, "CONTINUATION DETAILS", ["CONTINUATION", "DETAILS"], ["continuation_candidate"]),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 1
    assert "CONTINUATION DETAILS" in out[0].raw_text
    assert "row_stitched" in out[0].warnings
    assert out[0].metadata.get("stitched_fragments")
