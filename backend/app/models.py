from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    paper_name: str
    page_number: int
    chunk_id: str
    score: float


class PaperResponse(BaseModel):
    id: int
    name: str
    page_count: int
    chunk_count: int
    created_at: str


class CorpusStatusResponse(BaseModel):
    domain_name: str
    required_documents: int
    uploaded_documents: int
    remaining_documents: int
    total_pages: int
    total_chunks: int
    ready: bool


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    paper_id: int | None = Field(default=None, ge=1)
    session_id: str = Field(default="default", min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.:-]+$")


class CorrectiveAnalytics(BaseModel):
    retrieval_confidence: float
    correction_attempts: int
    final_confidence: float
    chunks_accepted: int
    chunks_rejected: int
    grounding_check_result: str
    chunk_grades: list[dict[str, str | float | int]]
    rewritten_query: str | None = None
    response_time_ms: int = 0


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    simple_explanation: str
    key_insights: list[str]
    confidence: float
    formatted: str
    analytics: Optional[CorrectiveAnalytics] = None


class SummaryResponse(BaseModel):
    paper_id: int
    summary: str
    citations: list[Citation]
    cached: bool = False


class SummaryRequest(BaseModel):
    paper_id: int = Field(..., ge=1)
    force: bool = False


class EvaluationQuestion(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    expected_answer_terms: list[str] = Field(default_factory=list)
    expected_paper_id: int | None = Field(default=None, ge=1)


class EvaluationRequest(BaseModel):
    questions: list[EvaluationQuestion] = Field(..., min_length=1, max_length=50)
    session_id: str = Field(default="evaluation", min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_.:-]+$")


class EvaluationResult(BaseModel):
    question: str
    answer: str
    retrieval_hit: bool
    answer_quality: float
    confidence: float
    matched_terms: list[str]
    citations: list[Citation]


class EvaluationResponse(BaseModel):
    retrieval_accuracy: float
    answer_quality: float
    total_questions: int
    results: list[EvaluationResult]


class ChatMessageResponse(BaseModel):
    id: int
    session_id: str
    paper_id: int | None
    role: str
    content: str
    created_at: str


class ErrorResponse(BaseModel):
    detail: str
