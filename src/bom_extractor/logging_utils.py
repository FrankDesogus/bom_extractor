from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(output_dir: Path) -> logging.Logger:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bom_extractor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = JsonLineFormatter()

    info_handler = logging.FileHandler(logs_dir / "pipeline.log.jsonl", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    error_handler = logging.FileHandler(logs_dir / "errors.log.jsonl", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger
