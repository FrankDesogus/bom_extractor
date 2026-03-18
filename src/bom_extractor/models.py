from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageContext(BaseModel):
    source_file: str
    source_file_hash: str
    page_number: int
    page_rotation: int = 0
    page_width: float = 0.0
    page_height: float = 0.0
    raw_page_text: str = ""
    layout_metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str


class RawRowRecord(BaseModel):
    source_file: str
    source_file_hash: str
    document_id: str
    page_number: int
    row_index_on_page: int
    raw_text: str
    extracted_columns: list[str] = Field(default_factory=list)
    item: str | None = None
    type_raw: str | None = None
    code: str | None = None
    revision: str | None = None
    description: str | None = None
    trade_name: str | None = None
    company_name: str | None = None
    uom: str | None = None
    quantity_raw: str | None = None
    notes: str | None = None
    parser_confidence: float = 0.0
    parser_name: str
    warnings: list[str] = Field(default_factory=list)
    bbox_row: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParserPageResult(BaseModel):
    parser_name: str
    page_number: int
    rows: list[RawRowRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParserScoreDetail(BaseModel):
    parser_name: str
    final_score: float
    row_count: int
    item_ratio: float
    quantity_ratio: float
    header_ratio: float
    footer_ratio: float
    fragmentation_penalty: float
    reasons: list[str] = Field(default_factory=list)


class PageFusionDecision(BaseModel):
    page_number: int
    selected_parser: str
    selected_score: float
    score_details: list[ParserScoreDetail] = Field(default_factory=list)
    disagreement: bool = False


class RowOutput(BaseModel):
    source_file: str
    page_number: int
    row_index: int
    raw_fragments: list[str] = Field(default_factory=list)
    reconstructed_text: str = ""
    extracted_columns: list[str] = Field(default_factory=list)
    parser_sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_band: str | None = None
    structural_state: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PageOutput(BaseModel):
    page_number: int
    page_state: str | None = None
    header_code: str | None = None
    header_revision: str | None = None
    header_type: str | None = None
    header_description: str | None = None
    header_fields_raw: list[str] = Field(default_factory=list)
    header_raw_lines: list[str] = Field(default_factory=list)
    header_bbox: tuple[float, float, float, float] | None = None
    header_zone_confidence: float | None = None
    footer_fields_raw: list[str] = Field(default_factory=list)
    reconstructed_table: list[RowOutput] = Field(default_factory=list)
    parser_decision: dict[str, Any] = Field(default_factory=dict)
    layout_model: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    source_file: str
    source_file_hash: str
    document_id: str
    pages_seen: int = 0
    rows_emitted: int = 0
    parser_usage: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    fusion_decisions: list[PageFusionDecision] = Field(default_factory=list)
    page_layouts: list[dict[str, Any]] = Field(default_factory=list)
    pages: list[PageOutput] = Field(default_factory=list)
    page_state_distribution: dict[str, int] = Field(default_factory=dict)
    structural_row_metrics: dict[str, Any] = Field(default_factory=dict)
    document_metadata: dict[str, Any] = Field(default_factory=dict)
