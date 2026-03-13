from bom_extractor.utils import looks_like_footer, looks_like_header, looks_like_item


def test_header_detection():
    text = 'Riga / Item Tipo / Type Codice / Code Rev. Descrizione / Description'
    assert looks_like_header(text)


def test_footer_detection():
    text = 'THIS DOCUMENT CONTAINS PROPRIETARY INFORMATION'
    assert looks_like_footer(text)


def test_item_detection():
    assert looks_like_item('0080')
    assert not looks_like_item('Minuteria')
