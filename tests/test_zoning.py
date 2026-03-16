from bom_extractor.zoning import infer_page_layout


def test_infer_page_layout_separates_header_table_footer():
    words = [
        (10.0, 20.0, 80.0, 28.0, "BILL"),
        (82.0, 20.0, 150.0, 28.0, "OF"),
        (152.0, 20.0, 240.0, 28.0, "MATERIAL"),
        (10.0, 100.0, 35.0, 108.0, "0010"),
        (40.0, 100.0, 120.0, 108.0, "PLATE"),
        (130.0, 100.0, 210.0, 108.0, "2"),
        (10.0, 112.0, 35.0, 120.0, "0020"),
        (40.0, 112.0, 120.0, 120.0, "BOLT"),
        (130.0, 112.0, 210.0, 120.0, "6"),
        (10.0, 760.0, 140.0, 768.0, "ISO 9001"),
        (145.0, 760.0, 260.0, 768.0, "CONFIDENTIAL"),
    ]

    layout = infer_page_layout(page_height=800.0, words=words)

    assert layout.header_lines
    assert any("BILL OF MATERIAL" in l for l in layout.header_lines)
    assert len(layout.table_lines) >= 2
    assert any("0010" in l for l in layout.table_lines)
    assert not layout.footer_lines
    assert layout.confidence > 0.4


def test_infer_page_layout_marks_background_noise():
    words = [
        (5.0, 20.0, 7.0, 27.0, "*"),
        (10.0, 100.0, 35.0, 108.0, "0010"),
        (40.0, 100.0, 120.0, 108.0, "PLATE"),
        (130.0, 100.0, 210.0, 108.0, "2"),
    ]
    layout = infer_page_layout(page_height=800.0, words=words)
    assert "*" in layout.background_noise_lines


def test_footer_capture_suspected_when_item_like_line_near_footer():
    words = [
        (10.0, 60.0, 80.0, 68.0, "RIGA"),
        (90.0, 60.0, 140.0, 68.0, "ITEM"),
        (150.0, 60.0, 190.0, 68.0, "CODE"),
        (10.0, 730.0, 35.0, 738.0, "0030"),
        (40.0, 730.0, 120.0, 738.0, "SUPPORT"),
    ]
    layout = infer_page_layout(page_height=800.0, words=words)
    assert any("0030" in l for l in layout.table_lines)
    assert "footer_capture_suspected" in layout.warnings
