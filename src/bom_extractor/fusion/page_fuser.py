from __future__ import annotations

from dataclasses import dataclass

from ..models import PageFusionDecision, ParserPageResult, ParserScoreDetail
from ..utils import looks_like_footer, looks_like_header, looks_like_item, looks_like_quantity


@dataclass(slots=True)
class PageResultFuser:
    low_confidence_threshold: float = 0.45

    def choose(self, page_number: int, parser_results: list[ParserPageResult]) -> tuple[ParserPageResult, PageFusionDecision]:
        if not parser_results:
            raise RuntimeError("No parser results available")

        score_details = [self._score_result(result) for result in parser_results]
        ranked = sorted(score_details, key=lambda d: d.final_score, reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        disagreement = bool(second and abs(best.final_score - second.final_score) < 0.05 and best.parser_name != second.parser_name)

        selected = next(result for result in parser_results if result.parser_name == best.parser_name)
        decision = PageFusionDecision(
            page_number=page_number,
            selected_parser=best.parser_name,
            selected_score=best.final_score,
            score_details=ranked,
            disagreement=disagreement,
        )
        return selected, decision

    def _score_result(self, result: ParserPageResult) -> ParserScoreDetail:
        rows = result.rows
        row_count = len(rows)
        if row_count == 0:
            return ParserScoreDetail(
                parser_name=result.parser_name,
                final_score=0.0,
                row_count=0,
                item_ratio=0.0,
                quantity_ratio=0.0,
                header_ratio=0.0,
                footer_ratio=0.0,
                fragmentation_penalty=0.0,
                reasons=["no_rows"],
            )

        item_hits = 0
        qty_hits = 0
        header_hits = 0
        footer_hits = 0
        fragmented_hits = 0
        for row in rows:
            first = row.extracted_columns[0] if row.extracted_columns else None
            if looks_like_item(first):
                item_hits += 1
            if any(looks_like_quantity(col) for col in row.extracted_columns):
                qty_hits += 1
            if looks_like_header(row.raw_text):
                header_hits += 1
            if looks_like_footer(row.raw_text):
                footer_hits += 1
            if len(row.extracted_columns) <= 2:
                fragmented_hits += 1

        item_ratio = item_hits / row_count
        quantity_ratio = qty_hits / row_count
        header_ratio = header_hits / row_count
        footer_ratio = footer_hits / row_count
        fragmentation_penalty = fragmented_hits / row_count

        final = (
            0.35 * result.confidence
            + 0.25 * item_ratio
            + 0.20 * quantity_ratio
            + 0.10 * min(row_count / 25, 1.0)
            - 0.20 * header_ratio
            - 0.15 * footer_ratio
            - 0.15 * fragmentation_penalty
            - 0.03 * len(result.errors)
        )

        reasons = [
            f"base_confidence={result.confidence:.2f}",
            f"item_ratio={item_ratio:.2f}",
            f"quantity_ratio={quantity_ratio:.2f}",
            f"header_ratio={header_ratio:.2f}",
            f"footer_ratio={footer_ratio:.2f}",
            f"fragmentation={fragmentation_penalty:.2f}",
        ]
        if final < self.low_confidence_threshold:
            reasons.append("low_final_score")

        return ParserScoreDetail(
            parser_name=result.parser_name,
            final_score=round(final, 4),
            row_count=row_count,
            item_ratio=round(item_ratio, 4),
            quantity_ratio=round(quantity_ratio, 4),
            header_ratio=round(header_ratio, 4),
            footer_ratio=round(footer_ratio, 4),
            fragmentation_penalty=round(fragmentation_penalty, 4),
            reasons=reasons,
        )
