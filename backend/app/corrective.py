from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document

from .config import FETCH_K
from .reranker import rerank_documents
from .vector_store import retrieve_documents

logger = logging.getLogger("research_summarizer")


@dataclass
class GradedChunk:
    document: Document
    score: float
    grade: str
    reason: str


@dataclass
class CorrectiveRetrieval:
    documents: list[Document]
    scores: list[float]
    first_confidence: float
    confidence: float
    grades: list[GradedChunk]
    correction_attempts: int
    rewritten_query: str | None


def corrective_retrieve(
    question: str,
    paper_id: int | None,
    rewrite_fn,
    session_id: str,
    include_references: bool = False,
) -> CorrectiveRetrieval:
    first = _retrieve_and_grade(question, paper_id, include_references, top_k=8, candidate_k=FETCH_K)
    confidence = _confidence(first)
    if _is_strong(first, confidence):
        accepted = _accepted(first)
        return CorrectiveRetrieval(
            documents=[item.document for item in accepted],
            scores=[item.score for item in accepted],
            first_confidence=confidence,
            confidence=confidence,
            grades=first,
            correction_attempts=0,
            rewritten_query=None,
        )

    rewritten = rewrite_fn(question, session_id, paper_id)
    logger.info("Corrective retry: first_confidence=%.3f rewritten_query=%r", confidence, rewritten)
    second_candidates = retrieve_documents(
        rewritten,
        paper_id,
        top_k=12,
        include_references=include_references,
        search_mode="hybrid",
        use_rerank=False,
        candidate_k=max(FETCH_K, 30),
    )
    reranked = rerank_documents(rewritten, second_candidates, limit=10) if second_candidates else []
    second = _grade_chunks(rewritten, reranked)
    second_confidence = _confidence(second)
    accepted = _accepted(second)
    return CorrectiveRetrieval(
        documents=[item.document for item in accepted],
        scores=[item.score for item in accepted],
        first_confidence=confidence,
        confidence=second_confidence,
        grades=second,
        correction_attempts=1,
        rewritten_query=rewritten if rewritten != question else None,
    )


def verify_and_correct_answer(answer: str, documents: list[Document]) -> tuple[str, str]:
    if not documents or not answer.strip():
        return "Information not found in uploaded research papers.", "failed_no_evidence"
    context = " ".join(document.page_content for document in documents)
    supported_lines: list[str] = []
    removed = 0
    for block in answer.splitlines():
        line = block.strip()
        if not line:
            supported_lines.append(block)
            continue
        if _is_placeholder_line(line):
            removed += 1
            continue
        if _is_structural_line(line) or _supported(line, context):
            supported_lines.append(block)
        else:
            removed += 1
            logger.info("Grounding correction removed unsupported line: %r", line[:180])

    corrected = "\n".join(supported_lines).strip()
    corrected = _drop_empty_heading_blocks(corrected)
    if not corrected:
        return "Information not found in uploaded research papers.", "failed_removed_unsupported"
    return corrected, "passed" if removed == 0 else "corrected"


def grades_for_analytics(grades: list[GradedChunk]) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for item in grades:
        metadata = item.document.metadata
        rows.append(
            {
                "chunk_id": str(metadata.get("chunk_id", "")),
                "page_number": int(metadata.get("page_number", 0)),
                "grade": item.grade,
                "score": round(float(item.score), 3),
                "reason": item.reason,
            }
        )
    return rows


def _retrieve_and_grade(question: str, paper_id: int | None, include_references: bool, top_k: int, candidate_k: int) -> list[GradedChunk]:
    chunks = retrieve_documents(
        question,
        paper_id,
        top_k=top_k,
        include_references=include_references,
        search_mode="hybrid",
        use_rerank=False,
        candidate_k=candidate_k,
    )
    return _grade_chunks(question, chunks)


def _grade_chunks(question: str, chunks: list[tuple[Document, float]]) -> list[GradedChunk]:
    graded: list[GradedChunk] = []
    query_terms = _terms(question)
    for document, score in chunks:
        text_terms = _terms(document.page_content)
        overlap = len(query_terms.intersection(text_terms))
        if score >= 3.0 or overlap >= 3:
            grade, reason = "relevant", "strong_score_or_term_overlap"
        elif score >= 0.12 or overlap >= 1:
            grade, reason = "partially relevant", "weak_but_usable_overlap"
        else:
            grade, reason = "irrelevant", "low_score_and_low_overlap"
        logger.info(
            "Retrieval grading: chunk_id=%s page=%s score=%.3f grade=%s reason=%s",
            document.metadata.get("chunk_id"),
            document.metadata.get("page_number"),
            score,
            grade,
            reason,
        )
        graded.append(GradedChunk(document, score, grade, reason))
    return graded


def _accepted(grades: list[GradedChunk]) -> list[GradedChunk]:
    return [item for item in grades if item.grade in {"relevant", "partially relevant"}][:5]


def _confidence(grades: list[GradedChunk]) -> float:
    accepted = _accepted(grades)
    if not accepted:
        return 0.0
    grade_bonus = {"relevant": 1.0, "partially relevant": 0.55}
    return sum(max(item.score, 0.0) * grade_bonus[item.grade] for item in accepted) / len(accepted)


def _is_strong(grades: list[GradedChunk], confidence: float) -> bool:
    accepted = _accepted(grades)
    relevant_count = sum(1 for item in accepted if item.grade == "relevant")
    return len(accepted) >= 2 and (confidence >= 0.15 or relevant_count >= 1)


def _terms(text: str) -> set[str]:
    stop = {"what", "does", "paper", "study", "about", "with", "from", "this", "that", "the", "and", "for", "are"}
    return {term for term in re.findall(r"[a-z0-9+-]{3,}", text.lower()) if term not in stop}


def _supported(line: str, context: str) -> bool:
    line_terms = _terms(line)
    if len(line_terms) <= 2:
        return True
    context_terms = _terms(context)
    overlap = len(line_terms.intersection(context_terms))
    return overlap >= max(2, min(5, len(line_terms) // 3))


def _is_structural_line(line: str) -> bool:
    return bool(re.match(r"^(#+\s+.+|[-*]\s*$|\d+\.\s*$|answer:|citations:|simple explanation:|key insights:)$", line.lower()))


def _drop_empty_heading_blocks(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_body
        if current_heading is None:
            return
        body = [line for line in current_body if line.strip() and not _is_placeholder_line(line)]
        if body:
            output.append(current_heading)
            output.extend(body)
        current_heading = None
        current_body = []

    for line in lines:
        if _is_heading(line):
            flush()
            current_heading = line
            current_body = []
        elif current_heading is None:
            if not _is_placeholder_line(line):
                output.append(line)
        else:
            current_body.append(line)
    flush()
    return "\n".join(output).strip()


def _is_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S", line.strip()) or re.match(r"^[A-Z][A-Za-z /-]{2,}:\s*$", line.strip()))


def _is_placeholder_line(line: str) -> bool:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip(".:;-").lower()
    patterns = [
        r"^not found in uploaded paper content$",
        r"^information not found in uploaded research papers$",
        r"^no information available$",
        r"^no information is available$",
        r"^no relevant information available$",
        r"^no relevant information is available$",
        r"^no retrieved evidence$",
        r"^no evidence available$",
        r"^not available$",
        r"^none$",
        r"^n/a$",
    ]
    return bool(cleaned and any(re.match(pattern, cleaned) for pattern in patterns))
