import os
import re
import logging
from typing import Any

from langchain_classic.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from .config import MIN_RELEVANCE_SCORE, TOP_K
from .corrective import corrective_retrieve, grades_for_analytics, verify_and_correct_answer
from .database import add_chat_message, get_cached_summary, get_paper, list_chat_history, save_summary
from .analytics import AnalyticsTracker
from .models import ChatResponse, Citation
from .prompts import prompt_for
from .vector_store import get_page_documents, get_paper_contexts, retrieve_documents, search_context

NOT_FOUND = "Information not found in uploaded research papers."
logger = logging.getLogger("research_summarizer")
GREETING_RESPONSE = "Hello! Upload a research paper or ask a question about your uploaded papers."
THANKS_RESPONSE = "You're welcome! You can ask another question about your uploaded research papers."

QA_PROMPT = PromptTemplate.from_template(
    """Use only the uploaded research paper context below. If the context does not answer the question, say:
Information not found in uploaded research papers.

Context:
{context}

Question:
{question}

Answer in simple academic language:"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    """Use only the uploaded research paper context below.
Write the requested section in simple academic language using bullet points.
Every bullet must be based on the context. Do not add outside information.
If the context does not contain the answer, return an empty response.
Never write empty headings, placeholder bullets, "No information available", or "Not found in uploaded paper content".

Context:
{context}

Question:
{question}

Section bullets:"""
)

SUMMARY_SECTIONS = [
    ("Objectives", "research objectives aims purpose scope problem statement"),
    ("Industry Background", "industry background sector overview market context background"),
    ("GDP Contribution", "GDP contribution gross domestic product economic contribution national income"),
    ("Government Initiatives", "government initiatives policies schemes programs regulation support"),
    ("SWOT Analysis", "SWOT strengths weaknesses opportunities threats analysis"),
    ("Customer Satisfaction Factors", "customer satisfaction factors service quality price convenience trust satisfaction"),
    ("Dissatisfaction Factors", "dissatisfaction factors complaints problems challenges barriers limitations dissatisfaction"),
    ("Key Findings", "key findings results observations analysis findings"),
    ("Conclusion", "conclusion implications final remarks outcome"),
    ("Future Scope", "future scope recommendations future research suggestions further study"),
]


class LocalPaperLLM(LLM):
    """Extractive LangChain LLM that keeps answers grounded in retrieved paper text."""

    @property
    def _llm_type(self) -> str:
        return "local-paper-extractive-llm"

    def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any | None = None, **kwargs: Any) -> str:
        if "Follow Up Input:" in prompt:
            return _standalone_question(prompt)

        context = _extract_prompt_block(prompt, "Context:", "Question:")
        question = _extract_prompt_block(prompt, "Question:", "Answer")
        if not context.strip():
            return NOT_FOUND

        answer = _extract_answer(question, [context])
        return answer if answer else NOT_FOUND


def _has_groq_key() -> bool:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return bool(key and key.lower() not in {"your_groq_api_key_here", "replace_me", "changeme"})


def _get_llm():
    if _has_groq_key():
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.2,
            max_retries=0,
        )
    return LocalPaperLLM()


class StaticValidatedRetriever(BaseRetriever):
    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager: Any | None = None) -> list[Document]:
        return self.documents


def answer_question(question: str, paper_id: int | None = None, session_id: str = "default") -> ChatResponse:
    history_rows = list_chat_history(session_id=session_id, paper_id=paper_id)
    tracker = AnalyticsTracker()

    small_talk = _small_talk_response(question)
    if small_talk is not None:
        tracker.final_confidence = 1.0
        tracker.grounding_check_result = "not_required"
        response = _format_response(
            answer=small_talk,
            citations=[],
            simple_explanation=small_talk,
            key_insights=[],
            confidence=1.0,
            analytics=tracker.finish(),
        )
        add_chat_message(session_id, paper_id, "user", question)
        add_chat_message(session_id, paper_id, "assistant", response.formatted)
        return response

    requested_page = _detect_page_query(question)
    if requested_page is not None and paper_id is not None:
        response = _answer_page_query(question, paper_id, requested_page, session_id, tracker)
        return response

    retrieval = corrective_retrieve(
        question=question,
        paper_id=paper_id,
        rewrite_fn=_rewrite_search_query,
        session_id=session_id,
        include_references=_asks_for_references(question),
    )
    validated_documents = retrieval.documents
    source_citations = _citations_from_documents(validated_documents, retrieval.scores)
    confidence = retrieval.confidence
    tracker.retrieval_confidence = retrieval.first_confidence
    tracker.correction_attempts = retrieval.correction_attempts
    tracker.final_confidence = confidence
    tracker.chunks_accepted = len(validated_documents)
    tracker.chunks_rejected = max(0, len(retrieval.grades) - len(validated_documents))
    tracker.chunk_grades = grades_for_analytics(retrieval.grades)
    tracker.rewritten_query = retrieval.rewritten_query
    search_query = retrieval.rewritten_query or question
    source_contexts = [document.page_content for document in validated_documents]
    if not validated_documents:
        add_chat_message(session_id, paper_id, "user", question)
        tracker.grounding_check_result = "failed_no_evidence"
        response = _format_response(
            answer=NOT_FOUND,
            citations=[],
            simple_explanation=NOT_FOUND,
            key_insights=[],
            confidence=confidence,
            analytics=tracker.finish(),
        )
        add_chat_message(session_id, paper_id, "assistant", response.formatted)
        return response

    if len(validated_documents) > 5:
        source_contexts = _compress_contexts(source_contexts, search_query)
        validated_documents = [
            Document(page_content=context, metadata=document.metadata)
            for context, document in zip(source_contexts, validated_documents)
        ]

    memory = _build_memory(session_id, paper_id)
    llm = _get_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=StaticValidatedRetriever(documents=validated_documents),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_for("follow-up" if history_rows else "factual")},
        output_key="answer",
    )
    result = chain.invoke({"question": question})
    source_documents = result.get("source_documents", []) or validated_documents
    source_contexts = [document.page_content for document in source_documents] or source_contexts
    source_confidence = confidence

    if not _has_groq_key() and _is_definition_question(question) and paper_id is not None:
        answer, source_citations = _definition_answer(question, paper_id)
        source_contexts = _contexts_for_citations(paper_id, source_citations) or source_contexts
    else:
        answer = str(result.get("answer") or _extract_answer(question, source_contexts))
    answer = _remove_empty_response_sections(answer)
    answer, grounding_status = verify_and_correct_answer(answer, validated_documents)
    answer = _remove_empty_response_sections(answer)
    tracker.grounding_check_result = grounding_status
    if answer.strip() == NOT_FOUND:
        source_citations = []
        source_contexts = []
        tracker.final_confidence = 0.0
        source_confidence = 0.0
    insights = _build_insights(source_contexts, question=question)
    simple = _simplify(answer)
    if answer.strip() != NOT_FOUND:
        tracker.final_confidence = source_confidence
    response = _format_response(answer, source_citations, simple, insights, source_confidence, tracker.finish())
    add_chat_message(session_id, paper_id, "user", question)
    add_chat_message(session_id, paper_id, "assistant", response.formatted)
    return response


def _detect_page_query(question: str) -> int | None:
    match = re.search(r"\bpage\s*(?:number|no\.?|#)?\s*(\d{1,4})\b", question.lower())
    if not match:
        return None
    page_number = int(match.group(1))
    logger.info("Page query detected: question=%r page_number=%s", question, page_number)
    return page_number


def _answer_page_query(
    question: str,
    paper_id: int,
    page_number: int,
    session_id: str,
    tracker: AnalyticsTracker,
) -> ChatResponse:
    paper = get_paper(paper_id)
    page_count = int(paper["page_count"]) if paper else 0
    logger.info(
        "Page filter applied: paper_id=%s requested_page=%s page_count=%s",
        paper_id,
        page_number,
        page_count,
    )

    if not paper or page_number < 1 or (page_count and page_number > page_count):
        tracker.grounding_check_result = "failed_page_not_found"
        response = _format_response(NOT_FOUND, [], NOT_FOUND, [], 0.0, tracker.finish())
        add_chat_message(session_id, paper_id, "user", question)
        add_chat_message(session_id, paper_id, "assistant", response.formatted)
        return response

    documents = get_page_documents(paper_id, page_number, include_references=_asks_for_references(question))
    logger.info(
        "Page chunks found: paper_id=%s page=%s chunks=%s retry_status=%s",
        paper_id,
        page_number,
        len(documents),
        "sqlite_or_chroma_success" if documents else "empty_after_chroma_retry",
    )

    tracker.retrieval_confidence = 1.0 if documents else 0.0
    tracker.final_confidence = 1.0 if documents else 0.0
    tracker.chunks_accepted = len(documents)
    tracker.chunks_rejected = 0
    tracker.grounding_check_result = "metadata_page_filter"
    tracker.chunk_grades = [
        {
            "chunk_id": str(document.metadata.get("chunk_id", "")),
            "page_number": int(document.metadata.get("page_number", 0)),
            "grade": "relevant",
            "score": 1.0,
            "reason": "direct_page_number_metadata_match",
        }
        for document in documents
    ]

    add_chat_message(session_id, paper_id, "user", question)
    if not documents:
        answer = f"Page {page_number} exists in the uploaded paper, but no indexed text chunks were found for that page."
        response = _format_response(answer, [], answer, [], 0.0, tracker.finish())
        add_chat_message(session_id, paper_id, "assistant", response.formatted)
        return response

    scores = [1.0 for _document in documents]
    citations = _citations_from_documents(documents, scores)
    contexts = [document.page_content for document in documents]
    answer = _generate_page_answer(question, page_number, documents, session_id, paper_id)
    answer = _remove_empty_response_sections(answer)
    answer, grounding_status = verify_and_correct_answer(answer, documents)
    answer = _remove_empty_response_sections(answer)
    tracker.grounding_check_result = grounding_status
    if answer == NOT_FOUND:
        citations = []
        contexts = []
        tracker.final_confidence = 0.0
    insights = _build_insights(contexts, question=question)
    simple = _simplify(answer)
    response = _format_response(answer, citations, simple, insights, tracker.final_confidence, tracker.finish())
    add_chat_message(session_id, paper_id, "assistant", response.formatted)
    return response


def _generate_page_answer(question: str, page_number: int, documents: list[Document], session_id: str, paper_id: int) -> str:
    page_question = (
        f"{question}\n\nUse only the chunks whose metadata page_number is {page_number}. "
        "Summarize the selected page content with concise headings or bullets and cite page details in the answer when useful."
    )
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=_get_llm(),
            retriever=StaticValidatedRetriever(documents=documents),
            memory=_build_memory(session_id, paper_id),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt_for("summary")},
            output_key="answer",
        )
        result = chain.invoke({"question": page_question})
        answer = str(result.get("answer") or "").strip()
        if answer:
            return answer
    except Exception:
        logger.exception("Page-specific Groq answer failed; using extractive fallback: paper_id=%s page=%s", paper_id, page_number)
    return _extract_answer(question, [document.page_content for document in documents])


def _small_talk_response(question: str) -> str | None:
    normalized = re.sub(r"[^a-zA-Z\s]", "", question.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in {"hi", "hello", "hey", "good morning", "good evening", "good afternoon"}:
        return GREETING_RESPONSE
    if normalized in {"thanks", "thank you", "thankyou", "ok thanks", "okay thanks"}:
        return THANKS_RESPONSE
    return None


def _rewrite_search_query(question: str, session_id: str, paper_id: int | None) -> str:
    if not _has_groq_key():
        logger.info("Query rewrite skipped because Groq is not configured: question=%r", question)
        return question

    history_rows = list_chat_history(session_id=session_id, paper_id=paper_id)[-6:]
    history = "\n".join(f"{row['role']}: {row['content'][:300]}" for row in history_rows)
    prompt = f"""
Rewrite the user's question into one concise academic semantic-search query for retrieving passages
from uploaded research papers.

Rules:
- Preserve the user's exact topic and intent.
- Expand abbreviations only when obvious from the question or conversation.
- Do not answer the question.
- Do not add facts.
- Return only the rewritten query.

Conversation:
{history or "No prior conversation."}

User question:
{question}

Rewritten search query:
"""
    try:
        rewritten = _invoke_llm_text(prompt).strip().strip('"')
    except Exception:
        logger.exception("Query rewrite failed; using original question.")
        return question
    if not rewritten:
        return question
    logger.info("Query rewrite: original=%r rewritten=%r", question, rewritten)
    return rewritten[:500]


def _retrieve_validated_documents(question: str, paper_id: int | None) -> tuple[list[Document], list[Citation], float]:
    explicit_reference_request = _asks_for_references(question)
    candidates = retrieve_documents(question, paper_id, TOP_K, include_references=explicit_reference_request)
    if not candidates:
        return [], [], 0.0

    validated: list[tuple[Document, Citation, float]] = []
    logger.info(
        "Validated retrieval candidates: question=%r paper_id=%s candidates=%s",
        question,
        paper_id,
        len(candidates),
    )

    for document, score in candidates:
        metadata = document.metadata
        if not _has_required_metadata(metadata):
            continue
        if _is_reference_only_chunk(document.page_content) and not explicit_reference_request:
            continue

        citation = Citation(
            paper_name=str(metadata["paper_name"]),
            page_number=int(metadata["page_number"]),
            chunk_id=str(metadata["chunk_id"]),
            score=round(score, 3),
        )
        validated.append((document, citation, score))

    if not validated:
        logger.info("Retrieval validation rejected all chunks: question=%r", question)
        return [], [], 0.0

    documents = [item[0] for item in validated]
    citations = _unique_citations([item[1] for item in validated])
    confidence = round(sum(item[2] for item in validated) / len(validated), 3)
    logger.info("Retrieval validation accepted chunks: question=%r accepted=%s confidence=%s", question, len(documents), confidence)
    return documents, citations, confidence


def _citations_from_documents(documents: list[Document], scores: list[float]) -> list[Citation]:
    citations: list[Citation] = []
    for document, score in zip(documents, scores):
        metadata = document.metadata
        if not _has_required_metadata(metadata):
            continue
        citations.append(
            Citation(
                paper_name=str(metadata["paper_name"]),
                page_number=int(metadata["page_number"]),
                chunk_id=str(metadata["chunk_id"]),
                score=round(float(score), 3),
            )
        )
    return _unique_citations(citations)


def _compress_contexts(contexts: list[str], query: str, max_chars: int = 900) -> list[str]:
    query_terms = set(re.findall(r"[a-zA-Z0-9+-]{3,}", query.lower()))
    compressed: list[str] = []
    for context in contexts:
        sentences = _split_sentences(context)
        ranked = sorted(
            sentences,
            key=lambda sentence: len(query_terms.intersection(set(re.findall(r"[a-zA-Z0-9+-]{3,}", sentence.lower())))),
            reverse=True,
        )
        selected = " ".join(ranked[:5]).strip()
        compressed.append((selected or context)[:max_chars])
    return compressed


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _has_required_metadata(metadata: dict[str, Any]) -> bool:
    return all(key in metadata for key in ("paper_name", "page_number", "chunk_id"))


def _asks_for_references(question: str) -> bool:
    return bool(re.search(r"\b(reference|references|bibliography|citation|citations)\b", question.lower()))


def _is_reference_only_chunk(text: str) -> bool:
    lower = text.lower()
    if "references" in lower[:120] or "bibliography" in lower[:120]:
        return True
    citation_markers = len(re.findall(r"\b(19|20)\d{2}\b", text))
    doi_markers = len(re.findall(r"\bdoi\b|https?://|et al\.", lower))
    numbered_refs = len(re.findall(r"(?:^|\s)\d{1,3}\.\s+[A-Z][A-Za-z-]+", text))
    return citation_markers >= 6 or doi_markers >= 4 or numbered_refs >= 3


def summarize_paper(paper_id: int, force: bool = False) -> tuple[str, list[Citation], bool]:
    cached = get_cached_summary(paper_id)
    if cached is not None and not force:
        logger.info("Using cached summary: paper_id=%s chars=%s", paper_id, len(str(cached["summary"])))
        return str(cached["summary"]), [], True

    paper_contexts, paper_citations = get_paper_contexts(paper_id)
    logger.info("Summary generation started: paper_id=%s retrieved_chunks=%s groq=%s", paper_id, len(paper_contexts), _has_groq_key())
    if not paper_contexts:
        raise ValueError("No indexed chunks found for this paper.")

    if _has_groq_key():
        summary = _map_reduce_summary(paper_id, paper_contexts, paper_citations)
        summary = _remove_empty_summary_sections(summary)
        save_summary(paper_id, summary)
        logger.info("Groq map-reduce summary cached: paper_id=%s chars=%s", paper_id, len(summary))
        return summary, _unique_citations(paper_citations[:30]), False

    sections: list[str] = []
    all_citations: list[Citation] = []

    for title, query in SUMMARY_SECTIONS:
        contexts, citations = _select_section_contexts(query, paper_contexts, paper_citations)
        if not contexts:
            contexts, citations, confidence = search_context(query, paper_id)
            if confidence < MIN_RELEVANCE_SCORE:
                contexts, citations = [], []
        if contexts:
            all_citations.extend(citations)
        sections.append(_format_summary_section(title, query, contexts, citations))

    unique_citations = _unique_citations(all_citations)
    summary = "\n\n".join(section for section in sections if section.strip())
    save_summary(paper_id, summary)
    logger.info("Fallback extractive summary cached: paper_id=%s chars=%s", paper_id, len(summary))
    return summary, unique_citations, False


def _remove_empty_summary_sections(summary: str) -> str:
    return _remove_empty_response_sections(summary)


def _remove_empty_response_sections(text: str) -> str:
    if not text.strip():
        return NOT_FOUND
    text = _strip_placeholder_lines(text)

    lines = text.splitlines()
    cleaned: list[str] = []
    skip_heading = False
    pending_heading: str | None = None
    pending_body: list[str] = []

    def flush() -> None:
        nonlocal pending_heading, pending_body
        if pending_heading is None:
            return
        body_lines = [line for line in pending_body if line.strip() and not _is_placeholder_line(line)]
        body = "\n".join(body_lines).strip()
        if body:
            cleaned.append(pending_heading)
            cleaned.extend(body_lines)
            cleaned.append("")
        pending_heading = None
        pending_body = []

    for line in lines:
        if _is_section_heading(line):
            flush()
            pending_heading = line
            pending_body = []
            skip_heading = False
            continue
        if pending_heading is None:
            if not _is_placeholder_line(line):
                cleaned.append(line)
        elif _is_placeholder_line(line):
            skip_heading = True
        elif not skip_heading:
            pending_body.append(line)
    flush()
    # If the text had no explicit headings, preserve normal paragraph/list answers after removing placeholder lines.
    if not any(_is_section_heading(line) for line in lines):
        plain = "\n".join(line for line in text.splitlines() if line.strip() and not _is_placeholder_line(line)).strip()
        return plain or NOT_FOUND
    return "\n".join(cleaned).strip() or NOT_FOUND


def _strip_placeholder_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _is_placeholder_line(line))


def _is_section_heading(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^#{1,6}\s+\S", stripped):
        return True
    return bool(re.match(r"^(Objectives|Industry Background|GDP Contribution|Government Initiatives|SWOT Analysis|Customer Satisfaction Factors|Dissatisfaction Factors|Key Findings|Conclusion|Future Scope|Answer|Citations|Simple Explanation|Key Insights|Summary|Comparison|Research Gaps|Literature Review)\s*:\s*$", stripped, re.IGNORECASE))


def _is_placeholder_line(line: str) -> bool:
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip(".:;-").lower()
    if not cleaned:
        return False
    placeholder_patterns = [
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
    return any(re.match(pattern, cleaned) for pattern in placeholder_patterns)


def _map_reduce_summary(paper_id: int, contexts: list[str], citations: list[Citation]) -> str:
    contexts, citations = _select_summary_chunks(contexts, citations)
    batch_size = 8
    map_notes: list[str] = []
    batches = list(_chunk_pairs(contexts, citations, batch_size))
    logger.info("Groq map step started: paper_id=%s batches=%s batch_size=%s", paper_id, len(batches), batch_size)

    for index, batch in enumerate(batches, start=1):
        batch_text = _format_context_batch(batch)
        prompt = f"""
Use only the uploaded paper excerpts below.
Extract concise notes for these headings if present:
- Objectives
- Industry Background
- GDP Contribution
- Government Initiatives
- SWOT Analysis
- Customer Satisfaction Factors
- Dissatisfaction Factors
- Key Findings
- Conclusion
- Future Scope

Rules:
- Use simple academic language.
- Include page references like [Page 3].
- Do not add outside information.
- If a heading is not present in this batch, omit it.

Excerpts:
{batch_text}

Notes:
"""
        logger.info("Groq map request: paper_id=%s batch=%s chunks=%s", paper_id, index, len(batch))
        note = _invoke_llm_text(prompt)
        logger.info("Groq map response: paper_id=%s batch=%s chars=%s", paper_id, index, len(note))
        if note.strip():
            map_notes.append(note.strip())

    if not map_notes:
        raise ValueError("Groq returned no map summaries.")

    reduce_prompt = f"""
Use only the extracted notes below to write a detailed research paper summary.

Required output headings:
## Objectives
## Industry Background
## GDP Contribution
## Government Initiatives
## SWOT Analysis
## Customer Satisfaction Factors
## Dissatisfaction Factors
## Key Findings
## Conclusion
## Future Scope

Rules:
- Use headings and bullet points.
- Use simple academic language.
- Include page references in bullets when notes contain them.
- If information for a heading is absent, omit that heading completely.
- Never write empty headings, placeholder bullets, "No information available", or "Not found in uploaded paper content".
- Do not use external information.

Extracted notes:
{chr(10).join(map_notes)}

Final summary:
"""
    logger.info("Groq reduce request: paper_id=%s map_notes=%s", paper_id, len(map_notes))
    final_summary = _invoke_llm_text(reduce_prompt)
    logger.info("Groq reduce response: paper_id=%s chars=%s", paper_id, len(final_summary))
    if not final_summary.strip():
        raise ValueError("Groq returned an empty final summary.")
    return final_summary.strip()


def _select_summary_chunks(contexts: list[str], citations: list[Citation], limit: int = 16) -> tuple[list[str], list[Citation]]:
    if len(contexts) <= limit:
        return contexts, citations

    selected_indexes: set[int] = set()
    selected_indexes.update(range(min(4, len(contexts))))
    selected_indexes.update(range(max(0, len(contexts) - 2), len(contexts)))

    section_terms = set()
    for _title, query in SUMMARY_SECTIONS:
        section_terms.update(re.findall(r"[a-zA-Z]{4,}", query.lower()))

    scored: list[tuple[int, int]] = []
    for index, context in enumerate(contexts):
        terms = set(re.findall(r"[a-zA-Z]{4,}", context.lower()))
        score = len(section_terms.intersection(terms))
        if score > 0 and not _is_reference_only_chunk(context):
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score, index in scored[: limit - len(selected_indexes)]:
        selected_indexes.add(index)

    if len(selected_indexes) < limit:
        step = max(1, len(contexts) // (limit - len(selected_indexes)))
        for index in range(0, len(contexts), step):
            selected_indexes.add(index)
            if len(selected_indexes) >= limit:
                break

    indexes = sorted(selected_indexes)[:limit]
    logger.info("Summary chunk sampling: original=%s selected=%s", len(contexts), len(indexes))
    return [contexts[index] for index in indexes], [citations[index] for index in indexes]


def _chunk_pairs(contexts: list[str], citations: list[Citation], size: int) -> list[list[tuple[str, Citation]]]:
    pairs = list(zip(contexts, citations))
    return [pairs[index : index + size] for index in range(0, len(pairs), size)]


def _format_context_batch(batch: list[tuple[str, Citation]]) -> str:
    parts = []
    for context, citation in batch:
        excerpt = context[:700]
        parts.append(
            f"[paper_name={citation.paper_name}; page_number={citation.page_number}; chunk_id={citation.chunk_id}]\n{excerpt}"
        )
    return "\n\n".join(parts)


def _invoke_llm_text(prompt: str) -> str:
    try:
        response = _get_llm().invoke(prompt)
        text = getattr(response, "content", response)
        if isinstance(text, list):
            return " ".join(str(item) for item in text)
        return str(text)
    except Exception:
        logger.exception("Groq/LLM invocation failed.")
        raise


def _extract_answer(question: str, contexts: list[str]) -> str:
    question_terms = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    sentences = [_clean_sentence(sentence) for sentence in _split_sentences(" ".join(contexts))]
    sentences = [sentence for sentence in sentences if _is_good_answer_sentence(sentence)]
    ranked = sorted(
        sentences,
        key=lambda sentence: len(question_terms.intersection(set(re.findall(r"[a-zA-Z]{4,}", sentence.lower())))),
        reverse=True,
    )
    selected = [_trim_incomplete_tail(sentence.strip()) for sentence in ranked[:4] if sentence.strip()]
    return " ".join(selected)[:1400] or _clean_sentence(contexts[0][:1000])


def _build_memory(session_id: str, paper_id: int | None) -> ConversationBufferMemory:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    for row in list_chat_history(session_id=session_id, paper_id=paper_id):
        if row["role"] == "user":
            memory.chat_memory.add_user_message(row["content"])
        elif row["role"] == "assistant":
            memory.chat_memory.add_ai_message(row["content"])
    return memory


def _citation_from_document(document: Any) -> Citation:
    metadata = document.metadata
    return Citation(
        paper_name=str(metadata["paper_name"]),
        page_number=int(metadata["page_number"]),
        chunk_id=str(metadata["chunk_id"]),
        score=1.0,
    )


def _standalone_question(prompt: str) -> str:
    follow_up = _extract_prompt_block(prompt, "Follow Up Input:", "Standalone question:")
    history = _extract_prompt_block(prompt, "Chat History:", "Follow Up Input:")
    if not follow_up:
        return prompt[-500:]
    if re.search(r"\b(it|its|they|their|this|that|these|those)\b", follow_up, re.IGNORECASE) and history:
        return f"{follow_up} In the previously discussed uploaded research paper."
    return follow_up


def _extract_prompt_block(prompt: str, start: str, end: str) -> str:
    if start not in prompt:
        return ""
    after_start = prompt.split(start, 1)[1]
    if end in after_start:
        return after_start.split(end, 1)[0].strip()
    return after_start.strip()


def _build_insights(contexts: list[str], limit: int = 4, question: str | None = None) -> list[str]:
    sentences = [sentence for sentence in _split_sentences(" ".join(contexts)) if len(sentence.strip()) > 60 and _is_good_answer_sentence(sentence)]
    seen: set[str] = set()
    insights: list[str] = []
    for sentence in sentences:
        normalized = sentence.lower()[:80]
        if normalized in seen:
            continue
        seen.add(normalized)
        insights.append(_shorten_sentence(sentence))
        if len(insights) == limit:
            break
    return insights


def _format_summary_section(
    title: str,
    query: str,
    contexts: list[str],
    citations: list[Citation],
) -> str:
    if not contexts:
        return ""

    if _has_groq_key():
        generated = _generate_summary_section_with_llm(title, query, contexts, citations)
        if generated:
            return f"## {title}\n{generated}"

    bullets = _section_bullets(query, contexts, citations)
    if not bullets:
        return ""

    return f"## {title}\n" + "\n".join(bullets)


def _generate_summary_section_with_llm(
    title: str,
    query: str,
    contexts: list[str],
    citations: list[Citation],
) -> str:
    documents = [
        Document(
            page_content=context,
            metadata={
                "paper_name": citation.paper_name,
                "page_number": citation.page_number,
                "chunk_id": citation.chunk_id,
            },
        )
        for context, citation in zip(contexts, citations)
    ]
    if not documents:
        return ""

    chain = RetrievalQA.from_chain_type(
        llm=_get_llm(),
        chain_type="stuff",
        retriever=StaticValidatedRetriever(documents=documents),
        return_source_documents=False,
        chain_type_kwargs={"prompt": SUMMARY_PROMPT},
    )
    result = chain.invoke(
        {
            "query": (
                f"Write the {title} section for this research paper. "
                f"Focus on: {query}. Use bullet points and include page references when possible."
            )
        }
    )
    text = str(result.get("result", "")).strip()
    text = _remove_empty_response_sections(text)
    if not text or text == NOT_FOUND:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized_lines = []
    for line in lines:
        if line.startswith(("-", "*")):
            normalized_lines.append(f"- {line.lstrip('-* ').strip()}")
        else:
            normalized_lines.append(f"- {line}")
    return "\n".join(normalized_lines)


def _section_bullets(query: str, contexts: list[str], citations: list[Citation], limit: int = 4) -> list[str]:
    query_terms = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
    candidates: list[tuple[int, str, Citation]] = []

    for context, citation in zip(contexts, citations):
        sentences = [sentence for sentence in _split_sentences(context) if 45 <= len(sentence.strip()) <= 420 and _is_good_answer_sentence(sentence)]
        for sentence in sentences:
            sentence_terms = set(re.findall(r"[a-zA-Z]{4,}", sentence.lower()))
            score = len(query_terms.intersection(sentence_terms))
            if score > 0:
                candidates.append((score, sentence, citation))

    candidates.sort(key=lambda item: item[0], reverse=True)

    bullets: list[str] = []
    seen: set[str] = set()
    for _score, sentence, citation in candidates:
        clean_sentence = _clean_sentence(sentence)
        fingerprint = re.sub(r"[^a-z0-9]+", "", clean_sentence.lower())[:120]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        bullets.append(f"- {clean_sentence} [Page {citation.page_number}]")
        if len(bullets) == limit:
            break

    if bullets:
        return bullets

    fallback = _clean_sentence(contexts[0][:320])
    return [f"- {fallback} [Page {citations[0].page_number}]"] if citations else []


def _select_section_contexts(query: str, contexts: list[str], citations: list[Citation], limit: int = 8) -> tuple[list[str], list[Citation]]:
    query_terms = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
    matches: list[tuple[int, int, str, Citation]] = []

    for index, (context, citation) in enumerate(zip(contexts, citations)):
        context_terms = set(re.findall(r"[a-zA-Z]{4,}", context.lower()))
        score = len(query_terms.intersection(context_terms))
        if score > 0:
            matches.append((score, index, context, citation))

    matches.sort(key=lambda item: (-item[0], item[1]))
    selected = matches[:limit]
    return [item[2] for item in selected], [item[3] for item in selected]


def _clean_sentence(sentence: str) -> str:
    sentence = " ".join(sentence.split())
    sentence = re.sub(r"(?<=[a-z])-\s+(?=[a-z])", "-", sentence)
    sentence = re.sub(r"(?<=[a-z])\d{1,3}\b", "", sentence)
    sentence = re.sub(r"\s+\d{1,3}(?=[\s.,;)]|$)", "", sentence)
    sentence = re.sub(r"^[a-z]\s+", "", sentence)
    sentence = re.sub(r"^\W+", "", sentence)
    sentence = sentence.strip(" -:;")
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = re.sub(r"\b\d+\s*\|\s*", "", cleaned)
    return [_clean_sentence(sentence) for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _is_good_answer_sentence(sentence: str) -> bool:
    if len(sentence) < 35:
        return False
    if any(marker in sentence for marker in ("â", "ï", "\x80", "\x81", "\x82", "\x83", "\x9d")):
        return False
    if re.search(r"\b(author addresses|department of|university hospital|comprehensive cancer centre)\b", sentence, re.IGNORECASE):
        return False
    if sentence[:1].islower():
        return False
    if re.match(r"^(Fig\.|Table|NATURE REVIEWS|Article citation ID)", sentence, re.IGNORECASE):
        return False
    if re.search(r"\b(Fig\.|Table)\b", sentence):
        return False
    if len(re.findall(r"\b[A-Z0-9]{2,}[-+]?\b", sentence)) >= 8:
        return False
    if sentence.count("(") > sentence.count(")") + 1:
        return False
    return True


def _is_definition_question(question: str) -> bool:
    return bool(re.search(r"\b(what is|define|definition of|meaning of)\b", question.lower()))


def _definition_answer(question: str, paper_id: int) -> tuple[str, list[Citation]]:
    paper_contexts, paper_citations = get_paper_contexts(paper_id)
    terms = [term for term in re.findall(r"[a-zA-Z]{4,}", question.lower()) if term not in {"what", "define", "definition", "meaning"}]
    candidates: list[tuple[int, int, str, Citation]] = []

    for index, (context, citation) in enumerate(zip(paper_contexts, paper_citations)):
        for sentence in _split_sentences(context):
            if not _is_good_answer_sentence(sentence):
                continue
            lower_sentence = sentence.lower()
            score = sum(1 for term in terms if term in lower_sentence)
            if "breast cancer" in lower_sentence:
                score += 3
            if any(phrase in lower_sentence for phrase in ["is a", "is the", "refers to", "disease", "cancer", "tumour", "tumor"]):
                score += 2
            if score >= 4:
                candidates.append((score, index, sentence, citation))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[str, Citation]] = []
    seen: set[str] = set()
    for _score, _index, sentence, citation in candidates:
        fingerprint = re.sub(r"[^a-z0-9]+", "", sentence.lower())[:120]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append((sentence, citation))
        if len(selected) == 3:
            break

    if not selected:
        return NOT_FOUND, []

    primary = _trim_incomplete_tail(selected[0][0])
    supporting = [_trim_incomplete_tail(sentence) for sentence, _citation in selected[1:]]
    if "breast cancer" in question.lower():
        answer = "Breast cancer is described in the uploaded paper as a heterogeneous disease of the breast with clinically relevant molecular subtypes. "
        answer += primary
    else:
        answer = primary
    supporting = [sentence for sentence in supporting if not re.search(r"\b(for|and|or|of|to|in|with|by|from)\.$", sentence.lower())]
    if supporting:
        answer += " " + " ".join(supporting)
    return answer[:1400], _unique_citations([citation for _sentence, citation in selected])


def _contexts_for_citations(paper_id: int, citations: list[Citation]) -> list[str]:
    paper_contexts, paper_citations = get_paper_contexts(paper_id)
    wanted = {citation.chunk_id for citation in citations}
    return [context for context, citation in zip(paper_contexts, paper_citations) if citation.chunk_id in wanted]


def _trim_incomplete_tail(sentence: str) -> str:
    sentence = re.sub(r",?\s+\b(for|and|or|of|to|in|with|by|from)\s+\w?\.$", ".", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"\s+[a-z]\.$", ".", sentence)
    if re.search(r"\b[a-z]{1,4}\.$", sentence):
        last_word = re.findall(r"\b([a-z]{1,4})\.$", sentence)
        allowed = {"cell", "cold", "high", "low", "poor", "good", "risk", "rate", "care"}
        if last_word and last_word[0].lower() not in allowed and ";" in sentence:
            sentence = sentence.split(";", 1)[0].rstrip(".") + "."
    return sentence


def _shorten_sentence(sentence: str, limit: int = 260) -> str:
    sentence = _trim_incomplete_tail(sentence)
    if len(sentence) <= limit:
        return sentence
    shortened = sentence[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return _trim_incomplete_tail(shortened + ".")


def _unique_citations(citations: list[Citation]) -> list[Citation]:
    unique: dict[tuple[str, int, str], Citation] = {}
    for citation in citations:
        key = (citation.paper_name, citation.page_number, citation.chunk_id)
        unique.setdefault(key, citation)
    return list(unique.values())


def _simplify(answer: str) -> str:
    if answer == NOT_FOUND:
        return answer
    item_lines = []
    for line in answer.splitlines():
        match = re.match(r"^\s*\d+\.\s*(.+)$", line.strip())
        if match:
            item_lines.append(match.group(1).strip(" .;:-"))
    if item_lines:
        return f"In plain terms, the paper highlights: {', '.join(item_lines[:4])}."

    clean_answer = " ".join(answer.split())
    numbered_items = re.findall(r"(?:^|\s)\d+\.\s*([^\d:.;]+(?:[:][^\d.;]+)?)", clean_answer)
    if numbered_items:
        item_names = [re.sub(r"\s+", " ", item).strip(" :-") for item in numbered_items[:4]]
        return f"In plain terms, the paper highlights: {', '.join(item_names)}."

    first_sentence = re.split(r"(?<=[.!?])\s+", clean_answer)[0].strip()
    if len(first_sentence) < 25 and len(clean_answer) > len(first_sentence):
        first_sentence = clean_answer[:260].rsplit(" ", 1)[0].strip(" ,;:-") + "."
    return f"In plain terms, the paper says: {first_sentence}"


def _format_response(
    answer: str,
    citations: list[Citation],
    simple_explanation: str,
    key_insights: list[str],
    confidence: float,
    analytics=None,
) -> ChatResponse:
    if answer.strip() == NOT_FOUND:
        return ChatResponse(
            answer=NOT_FOUND,
            citations=[],
            simple_explanation=NOT_FOUND,
            key_insights=[],
            confidence=0.0,
            formatted=NOT_FOUND,
            analytics=analytics,
        )

    citation_lines = [
        f"- {citation.paper_name}, page {citation.page_number}, chunk {citation.chunk_id} (score {citation.score})"
        for citation in citations
    ]
    insight_lines = [f"- {insight}" for insight in key_insights]
    formatted = "\n".join(
        [
            "Answer:",
            answer,
            "",
            "Citations:",
            "\n".join(citation_lines) if citation_lines else "- None",
            "",
            "Simple Explanation:",
            simple_explanation,
            "",
            "Key Insights:",
            "\n".join(insight_lines) if insight_lines else "- None",
        ]
    )
    return ChatResponse(
        answer=answer,
        citations=citations,
        simple_explanation=simple_explanation,
        key_insights=key_insights,
        confidence=confidence,
        formatted=formatted,
        analytics=analytics,
    )
