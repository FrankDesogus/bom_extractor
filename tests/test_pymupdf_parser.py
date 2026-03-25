from pathlib import Path

from bom_extractor.models import PageContext
from bom_extractor.parsers.pymupdf_parser import PyMuPDFWordsParser


class _FakePage:
    def __init__(self, words, height=1000):
        self._words = words
        self.rect = type("Rect", (), {"height": height})()

    def get_text(self, kind):
        assert kind == "words"
        return self._words


class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        return None


def _ctx(page_number: int) -> PageContext:
    return PageContext(
        source_file="sample.pdf",
        source_file_hash="abc",
        page_number=page_number,
        document_id="doc-1",
        layout_metadata={"zone_header_cutoff": 80.0, "zone_footer_cutoff": 920.0},
    )


def test_footer_cutoff_preserves_last_clustered_row_and_drops_footer_text(monkeypatch):
    words = [
        (20, 700, 45, 710, "0155", 0, 0, 0),
        (90, 700, 130, 710, "TYPE", 0, 0, 1),
        (190, 700, 260, 710, "E0216160", 0, 0, 2),
        (300, 700, 320, 710, "01", 0, 0, 3),
        (355, 700, 370, 710, "02", 0, 0, 4),
        # Last valid table row sits close to footer and trips y1 >= footer_cutoff.
        (20, 914, 45, 923, "0165", 0, 0, 0),
        (90, 914, 130, 923, "TYPE", 0, 0, 1),
        (190, 914, 260, 923, "E0312345", 0, 0, 2),
        (300, 914, 320, 923, "01", 0, 0, 3),
        (355, 914, 370, 923, "03", 0, 0, 4),
        # Legal footer paragraph: broad continuous geometry, should still be removed.
        (20, 938, 85, 948, "CONFIDENTIAL", 0, 0, 0),
        (88, 938, 125, 948, "LEGAL", 0, 0, 1),
        (128, 938, 170, 948, "NOTICE", 0, 0, 2),
        (173, 938, 220, 948, "APPLIES", 0, 0, 3),
        (223, 938, 260, 948, "TO", 0, 0, 4),
        (263, 938, 320, 948, "THIS", 0, 0, 5),
        (323, 938, 385, 948, "DRAWING", 0, 0, 6),
    ]

    monkeypatch.setattr(
        "bom_extractor.parsers.pymupdf_parser.fitz.open",
        lambda _path: _FakeDoc([_FakePage(words)]),
    )

    parser = PyMuPDFWordsParser()
    result = parser.parse_page(Path("sample.pdf"), _ctx(page_number=1))

    row_texts = [r.raw_text for r in result.rows]
    assert any("0165" in t for t in row_texts)
    assert not any("CONFIDENTIAL" in t for t in row_texts)


def test_footer_override_does_not_regress_multi_page_parsing(monkeypatch):
    page1_words = [
        (20, 700, 45, 710, "0100", 0, 0, 0),
        (90, 700, 130, 710, "TYPE", 0, 0, 1),
        (190, 700, 260, 710, "E0100001", 0, 0, 2),
        (300, 700, 320, 710, "01", 0, 0, 3),
        (355, 700, 370, 710, "01", 0, 0, 4),
    ]
    page2_words = [
        (20, 910, 45, 923, "0200", 0, 0, 0),
        (90, 910, 130, 923, "TYPE", 0, 0, 1),
        (190, 910, 260, 923, "E0200002", 0, 0, 2),
        (300, 910, 320, 923, "01", 0, 0, 3),
        (355, 910, 370, 923, "02", 0, 0, 4),
        (20, 940, 90, 950, "FOOTER", 0, 0, 0),
        (93, 940, 150, 950, "PARAGRAPH", 0, 0, 1),
        (153, 940, 215, 950, "TEXT", 0, 0, 2),
    ]

    monkeypatch.setattr(
        "bom_extractor.parsers.pymupdf_parser.fitz.open",
        lambda _path: _FakeDoc([_FakePage(page1_words), _FakePage(page2_words)]),
    )

    parser = PyMuPDFWordsParser()
    page1 = parser.parse_page(Path("sample.pdf"), _ctx(page_number=1))
    page2 = parser.parse_page(Path("sample.pdf"), _ctx(page_number=2))

    assert any("0100" in r.raw_text for r in page1.rows)
    assert any("0200" in r.raw_text for r in page2.rows)
    assert not any("FOOTER" in r.raw_text for r in page2.rows)
