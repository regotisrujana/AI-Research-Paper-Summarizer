from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .models import CorrectiveAnalytics


@dataclass
class AnalyticsTracker:
    retrieval_confidence: float = 0.0
    correction_attempts: int = 0
    final_confidence: float = 0.0
    chunks_accepted: int = 0
    chunks_rejected: int = 0
    grounding_check_result: str = "not_checked"
    chunk_grades: list[dict[str, str | float | int]] = field(default_factory=list)
    rewritten_query: str | None = None

    def __post_init__(self) -> None:
        self._started = perf_counter()

    def finish(self) -> CorrectiveAnalytics:
        return CorrectiveAnalytics(
            retrieval_confidence=round(self.retrieval_confidence, 3),
            correction_attempts=self.correction_attempts,
            final_confidence=round(self.final_confidence, 3),
            chunks_accepted=self.chunks_accepted,
            chunks_rejected=self.chunks_rejected,
            grounding_check_result=self.grounding_check_result,
            chunk_grades=self.chunk_grades,
            rewritten_query=self.rewritten_query,
            response_time_ms=int((perf_counter() - self._started) * 1000),
        )
