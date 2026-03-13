from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import DocumentSummary, RawRowRecord


class StorageManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(self, rows: list[RawRowRecord]) -> Path:
        path = self.output_dir / "rows.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.model_dump(), ensure_ascii=False, default=str) + "\n")
        return path

    def write_csv(self, rows: list[RawRowRecord]) -> Path:
        path = self.output_dir / "rows.csv"
        df = pd.DataFrame([r.model_dump() for r in rows])
        df.to_csv(path, index=False)
        return path

    def write_parquet(self, rows: list[RawRowRecord]) -> Path | None:
        path = self.output_dir / "rows.parquet"
        df = pd.DataFrame([r.model_dump() for r in rows])
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            return None

    def write_summary(self, summary: DocumentSummary) -> Path:
        path = self.output_dir / "document_summary.json"
        path.write_text(json.dumps(summary.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path
