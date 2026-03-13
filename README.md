# BOM PDF Extractor - Phase 1

Industrial-grade, **lossless-first** PDF BOM row extraction for technical drawings.

## Scope (Phase 1)

This project focuses strictly on extraction robustness:
- multi-page BOM table extraction
- parser comparison and page-level selection
- dynamic layout-aware page zoning (header/table/footer)
- header/footer contamination reduction without silent row dropping
- multiline row stitching with traceability
- soft validation and warnings (never hard semantic rejection)
- batch processing and structured logging

It intentionally does **not** build BOM hierarchy or perform semantic classification.

## Architecture

```text
src/bom_extractor/
  cli.py
  config.py
  models.py
  pipeline.py
  utils.py
  logging_utils.py
  normalizer.py
  validators.py
  storage.py
  parsers/
    base.py
    camelot_parser.py
    pdfplumber_parser.py
    pymupdf_parser.py
    ocr_parser.py
  fusion/
    page_fuser.py
  zoning/
    page_zoning.py
  normalization/
    row_reconstruction.py
```

### Extraction flow

1. **Ingestion**
   - input file or folder of PDFs
   - per-document hash and context metadata
2. **Multi-parser extraction per page**
   - Camelot (lattice + stream fallback)
   - pdfplumber (line/text table strategies)
   - PyMuPDF word clustering
   - OCR fallback (optional stub)
3. **Layout inference (per page)**
   - infer dynamic `header_zone`, `table_zone`, `footer_zone` from geometry + lexical/table signals
   - retain `header_fields_raw`, `footer_fields_raw`, cutoffs, and layout warnings in summary
4. **Fusion / ranking**
   - explainable page score per parser
   - metrics: item ratio, quantity ratio, header/footer contamination, fragmentation, parser confidence
5. **Lossless normalization**
   - always preserve `raw_text` and `extracted_columns`
   - weak field projection (`item`, `code`, `uom`, `quantity_raw`, ...)
6. **Row reconstruction**
   - continuation stitching
   - stitched fragments kept in metadata for provenance
7. **Soft validation**
   - warnings for suspicious/header/footer/column-count anomalies
8. **Storage & logs**
   - JSONL primary output
   - CSV optional
   - Parquet optional
   - structured logs + document summary with fusion decisions

## Row model (mandatory fields)

Each output row includes:
- `source_file`
- `page_number`
- `row_index_on_page`
- `raw_text`
- `extracted_columns`
- `item`
- `type_raw`
- `code`
- `revision`
- `description`
- `uom`
- `quantity_raw`
- `notes`
- `parser_confidence`
- `parser_name`
- `warnings`

Additional provenance is preserved in `metadata`.

## CLI usage

```bash
python -m bom_extractor.cli parse --input "E0181296 01-06_BOM.pdf" --output-dir ./out
```

Batch mode:

```bash
python -m bom_extractor.cli parse --input ./pdf_folder --output-dir ./out
```

Optional OCR flag:

```bash
python -m bom_extractor.cli parse --input ./pdf_folder --output-dir ./out --enable-ocr
```

## Outputs

- `rows.jsonl` (primary)
- `rows.csv` (optional)
- `rows.parquet` (optional, if parquet engine available)
- `document_summary.json` (includes parser usage + fusion decisions + page-level layout/header/footer evidence)
- `logs/pipeline.log.jsonl`
- `logs/errors.log.jsonl`

## Testing

```bash
pytest -q
```

Tests cover smoke extraction, parser fusion behavior, stitching traceability, and utility heuristics.

## Current limitations

- OCR is still an optional placeholder and not a full OCR extraction stack.
- Heuristic scoring is explainable but not learned.
- Page zoning is conservative and geometry-driven.

## Next priorities

- Optional real OCR backend and parser integration.
- Adaptive zoning using repeated patterns across page sets.
- Better parser merge (not only winner-takes-page) with disagreement reconciliation.
