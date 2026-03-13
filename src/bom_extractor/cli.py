from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExtractionConfig
from .pipeline import ExtractionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BOM PDF extractor phase 1")
    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="Parse one PDF or a directory of PDFs")
    parse_cmd.add_argument("--input", required=True, help="PDF file or folder")
    parse_cmd.add_argument("--output-dir", required=True, help="Output directory")
    parse_cmd.add_argument("--enable-ocr", action="store_true")
    parse_cmd.add_argument("--disable-csv", action="store_true")
    parse_cmd.add_argument("--disable-parquet", action="store_true")
    parse_cmd.add_argument("--max-pages", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "parse":
        config = ExtractionConfig(
            output_dir=Path(args.output_dir),
            enable_ocr=args.enable_ocr,
            write_csv=not args.disable_csv,
            write_parquet=not args.disable_parquet,
            max_pages=args.max_pages,
        )
        pipeline = ExtractionPipeline(config)
        pipeline.parse_input(Path(args.input))


if __name__ == "__main__":
    main()
