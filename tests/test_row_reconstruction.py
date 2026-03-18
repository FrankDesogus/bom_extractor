from bom_extractor.models import RawRowRecord
from bom_extractor.normalization import stitch_multiline_rows
from bom_extractor.normalization.row_reconstruction import apply_page_lane_inference


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


def test_short_reference_fragment_routes_to_notes():
    rows = [
        _row(1, "0010 TYPE E0181296 01 NR 2", ["0010"], bbox=(10, 100, 220, 108), item="0010", code="E0181296", revision="01"),
        _row(2, "10_35", ["10_35"], ["continuation_candidate"], bbox=(160, 111, 210, 119)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 1
    assert out[0].notes
    assert "10_35" in out[0].notes


def test_continuation_does_not_cross_later_anchor_row():
    rows = [
        _row(1, "0010 FIRST", ["0010", "FIRST"], bbox=(10, 100, 220, 108), item="0010", code="E0181296", revision="01"),
        _row(2, "0020 SECOND", ["0020", "SECOND"], bbox=(10, 112, 220, 120), item="0020", code="E0181297", revision="01"),
        _row(3, "supplier BETA SRL", ["supplier", "BETA", "SRL"], ["continuation_candidate"], bbox=(100, 123, 240, 131)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2
    assert out[0].item == "0010"
    assert out[1].item == "0020"
    assert out[1].company_name and "BETA" in out[1].company_name


def test_orphan_continuation_fragment_is_preserved_with_warning():
    rows = [
        _row(1, "0010 BASE", ["0010", "BASE"], bbox=(10, 100, 120, 108), item="0010", code="E0181296", revision="01"),
        _row(2, "supplier ACME SRL", ["supplier", "ACME", "SRL"], ["continuation_candidate"], bbox=(350, 150, 450, 158)),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2
    assert "orphan_continuation_fragment" in out[1].warnings
    assert "parent_row_uncertain" in out[1].warnings


def test_lane_inference_sets_model_and_keeps_uom_qty_split():
    rows = [
        _row(
            1,
            "0010 TYPE E0181296 01 NR 2",
            ["0010", "TYPE", "E0181296", "01", "NR", "2"],
            bbox=(10, 100, 250, 108),
            metadata={
                "word_boxes": [
                    {"text": "0010", "x0": 10, "x1": 30},
                    {"text": "TYPE", "x0": 45, "x1": 80},
                    {"text": "E0181296", "x0": 95, "x1": 150},
                    {"text": "01", "x0": 165, "x1": 178},
                    {"text": "NR", "x0": 195, "x1": 210},
                    {"text": "2", "x0": 225, "x1": 232},
                ]
            },
        ),
        _row(
            2,
            "0020 TYPE E0181297 02 KG 4",
            ["0020", "TYPE", "E0181297", "02", "KG", "4"],
            bbox=(10, 112, 250, 120),
            metadata={
                "word_boxes": [
                    {"text": "0020", "x0": 10, "x1": 30},
                    {"text": "TYPE", "x0": 45, "x1": 80},
                    {"text": "E0181297", "x0": 95, "x1": 150},
                    {"text": "02", "x0": 165, "x1": 178},
                    {"text": "KG", "x0": 195, "x1": 210},
                    {"text": "4", "x0": 225, "x1": 232},
                ]
            },
        ),
    ]
    out, metrics = apply_page_lane_inference(rows)
    assert metrics["lane_count"] >= 5
    assert metrics["lane_confidence_score"] > 0
    assert out[0].metadata.get("page_lane_model")
    assert out[0].uom != out[0].quantity_raw
    assert out[1].uom != out[1].quantity_raw


def test_complete_anchor_row_not_marked_as_continuation_candidate():
    rows = [
        _row(1, "0010 TYPE E0181296 01 NR 2", ["0010", "TYPE", "E0181296", "01", "NR", "2"], bbox=(10, 100, 220, 108), item="0010", code="E0181296", revision="01", uom="NR", quantity_raw="2"),
        _row(2, "0020 TYPE E0181297 01 NR 3", ["0020", "TYPE", "E0181297", "01", "NR", "3"], ["continuation_candidate"], bbox=(10, 112, 220, 120), item="0020", code="E0181297", revision="01", uom="NR", quantity_raw="3"),
    ]
    out = stitch_multiline_rows(rows)
    assert len(out) == 2
    assert "continuation_candidate" not in out[1].warnings


def test_clean_anchor_row_denoises_soft_warnings_and_gets_high_confidence():
    rows = [
        _row(
            1,
            "0010 TYPE E0181296 01 NR 2",
            ["0010", "TYPE", "E0181296", "01", "NR", "2"],
            ["continuation_candidate", "field_assignment_uncertain", "lane_ambiguity"],
            bbox=(10, 100, 250, 108),
            item="0010",
            code="E0181296",
            revision="01",
            uom="NR",
            quantity_raw="2",
        ),
        _row(
            2,
            "0020 TYPE E0181297 01 NR 3",
            ["0020", "TYPE", "E0181297", "01", "NR", "3"],
            bbox=(10, 112, 250, 120),
            item="0020",
            code="E0181297",
            revision="01",
            uom="NR",
            quantity_raw="3",
        ),
    ]
    out, metrics = apply_page_lane_inference(rows)
    assert "continuation_candidate" not in out[0].warnings
    assert "field_assignment_uncertain" not in out[0].warnings
    assert "lane_ambiguity" not in out[0].warnings
    assert out[0].metadata["row_structure_classification"] == "clean_anchor_row"
    assert out[0].metadata["operational_confidence_band"] == "high"
    assert metrics["clean_anchor_row_count"] >= 2
    assert metrics["high_confidence_row_count"] >= 1
    assert metrics["noisy_warning_suppression_count"] >= 1


def test_attached_continuation_row_is_medium_and_ambiguous_row_is_low_confidence():
    rows = [
        _row(
            1,
            "0010 TYPE E0181296 01 NR 2 details",
            ["0010", "TYPE", "E0181296", "01", "NR", "2", "details"],
            bbox=(10, 100, 250, 108),
            item="0010",
            code="E0181296",
            revision="01",
            metadata={"stitched_fragments": [{"raw_text": "details"}]},
        ),
        _row(
            2,
            "bad row",
            ["bad", "row"],
            ["lane_ambiguity", "anchor_lane_conflict"],
            bbox=(10, 112, 140, 120),
        ),
    ]
    out, metrics = apply_page_lane_inference(rows)
    assert out[0].metadata["row_structure_classification"] == "anchor_row_with_expandable_continuation"
    assert out[0].metadata["operational_confidence_band"] == "medium"
    assert out[1].metadata["row_structure_classification"] == "ambiguous_row"
    assert out[1].metadata["operational_confidence_band"] == "low"
    assert metrics["continuation_row_count"] >= 1
    assert metrics["medium_confidence_row_count"] >= 1
    assert metrics["low_confidence_row_count"] >= 1


def test_orphan_fragment_row_gets_continuation_fragment_state():
    rows = [
        _row(1, "0010 TYPE E0181296 01 NR 2", ["0010"], bbox=(10, 100, 220, 108), item="0010", code="E0181296", revision="01"),
        _row(2, "supplier ACME SRL", ["supplier", "ACME", "SRL"], ["continuation_candidate"], bbox=(180, 112, 260, 120)),
    ]
    out, metrics = apply_page_lane_inference(rows)
    assert out[1].metadata["row_structure_classification"] == "continuation_fragment_row"
    assert out[1].metadata["operational_confidence_band"] == "low"
    assert "orphan_continuation_fragment" in out[1].warnings
    assert metrics["continuation_fragment_count"] >= 1


def test_header_and_footer_rows_are_structural_non_bom_states():
    rows = [
        _row(1, "ITEM CODE QTY DESCRIPTION", ["ITEM", "CODE", "QTY", "DESCRIPTION"], ["header_row", "probable_header_leakage"], bbox=(10, 80, 220, 88), metadata={"atomic_line": {"is_header_like": True, "starts_with_item_anchor": False}}),
        _row(2, "Rev. date 2024", ["Rev.", "date", "2024"], ["footer_row"], bbox=(10, 220, 120, 228), metadata={"atomic_line": {"is_footer_like": True}}),
    ]
    out, metrics = apply_page_lane_inference(rows)
    assert out[0].metadata["row_structure_classification"] == "table_header_row"
    assert out[1].metadata["row_structure_classification"] == "non_bom_structural_row"
    assert metrics["table_header_row_count"] == 1
    assert metrics["non_bom_structural_row_count"] == 1
