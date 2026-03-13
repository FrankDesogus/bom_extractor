from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ExtractionConfig:
    output_dir: Path
    enable_camelot: bool = True
    enable_pdfplumber: bool = True
    enable_pymupdf: bool = True
    enable_ocr: bool = False
    write_csv: bool = True
    write_parquet: bool = True
    max_pages: int | None = None
    continue_on_error: bool = True
    low_confidence_threshold: float = 0.45
    min_words_for_text_pdf: int = 20
    parser_order: list[str] = field(
        default_factory=lambda: [
            "camelot_lattice",
            "pdfplumber_table",
            "pymupdf_words",
            "ocr_fallback",
        ]
    )
