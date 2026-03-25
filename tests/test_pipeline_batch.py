from pathlib import Path

from bom_extractor.config import ExtractionConfig
from bom_extractor.models import DocumentSummary, RawRowRecord
from bom_extractor.pipeline import ExtractionPipeline


def _fake_row(source_file: str) -> RawRowRecord:
    return RawRowRecord(
        source_file=source_file,
        source_file_hash="hash",
        document_id="doc",
        page_number=1,
        row_index_on_page=1,
        raw_text="row",
        extracted_columns=["row"],
        parser_name="fake",
    )


def _fake_summary(source_file: str) -> DocumentSummary:
    return DocumentSummary(source_file=source_file, source_file_hash="hash", document_id="doc", rows_emitted=1)


def test_parse_input_directory_processes_only_bom_pdfs_recursively(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)

    bom_root = input_dir / "A_BOM.pdf"
    bom_nested = nested / "BOM_123.pdf"
    non_bom = nested / "other.pdf"
    text_file = input_dir / "notes.txt"

    bom_root.write_bytes(b"%PDF-1.4")
    bom_nested.write_bytes(b"%PDF-1.4")
    non_bom.write_bytes(b"%PDF-1.4")
    text_file.write_text("ignore", encoding="utf-8")

    config = ExtractionConfig(output_dir=tmp_path / "out", write_parquet=False)
    pipeline = ExtractionPipeline(config)

    parsed_paths: list[Path] = []

    def fake_parse_document(pdf_path: Path):
        parsed_paths.append(pdf_path)
        return [_fake_row(str(pdf_path))], _fake_summary(str(pdf_path))

    csv_calls: list[tuple[int, str]] = []

    def fake_write_csv(rows, filename="rows.csv"):
        csv_calls.append((len(rows), filename))
        return pipeline.storage.output_dir / filename

    monkeypatch.setattr(pipeline, "parse_document", fake_parse_document)
    monkeypatch.setattr(pipeline.storage, "write_csv", fake_write_csv)

    rows = pipeline.parse_input(input_dir)

    assert sorted(path.name for path in parsed_paths) == ["A_BOM.pdf", "BOM_123.pdf"]
    assert sorted(call[1] for call in csv_calls) == ["A_BOM.csv", "BOM_123.csv"]
    assert len(rows) == 2


def test_parse_input_directory_logs_and_continues_on_document_error(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)

    ok_pdf = input_dir / "ok_BOM.pdf"
    bad_pdf = input_dir / "bad_BOM.pdf"
    ok_pdf.write_bytes(b"%PDF-1.4")
    bad_pdf.write_bytes(b"%PDF-1.4")

    config = ExtractionConfig(output_dir=tmp_path / "out", write_parquet=False)
    pipeline = ExtractionPipeline(config)

    def fake_parse_document(pdf_path: Path):
        if pdf_path.name == "bad_BOM.pdf":
            raise RuntimeError("boom")
        return [_fake_row(str(pdf_path))], _fake_summary(str(pdf_path))

    csv_calls: list[str] = []

    def fake_write_csv(rows, filename="rows.csv"):
        csv_calls.append(filename)
        return pipeline.storage.output_dir / filename

    monkeypatch.setattr(pipeline, "parse_document", fake_parse_document)
    monkeypatch.setattr(pipeline.storage, "write_csv", fake_write_csv)

    rows = pipeline.parse_input(input_dir)

    assert len(rows) == 1
    assert csv_calls == ["ok_BOM.csv"]
