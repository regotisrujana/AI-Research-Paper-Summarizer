from pathlib import Path
from threading import Thread
from uuid import uuid4
import logging
import os

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import ALLOWED_EXTENSIONS, DATA_DIR, MAX_UPLOAD_SIZE_MB, MIN_SOURCE_DOCUMENTS, SOURCE_DOCS_DIR, UPLOAD_DIR
from .database import count_paper_embeddings, create_paper, delete_chat_history, delete_paper_record, get_cached_summary, get_paper, init_db, list_chat_history, list_papers, update_paper_chunks
from .models import ChatMessageResponse, ChatRequest, ChatResponse, CorpusStatusResponse, ErrorResponse, EvaluationRequest, EvaluationResponse, EvaluationResult, PaperResponse, SummaryRequest, SummaryResponse
from .pdf_processing import extract_pdf_chunks

DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DATA_DIR / "backend.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("research_summarizer")
_ingest_thread_started = False

app = FastAPI(
    title="AI Research Paper Summarizer API",
    description="Upload PDFs, store paper vectors in ChromaDB, and ask citation-grounded questions.",
    version="1.0.0",
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *([os.getenv("FRONTEND_ORIGIN", "").rstrip("/")] if os.getenv("FRONTEND_ORIGIN") else []),
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    _start_background_corpus_ingest()


def _start_background_corpus_ingest() -> None:
    global _ingest_thread_started
    if _ingest_thread_started:
        return
    if os.getenv("AUTO_INGEST_CORPUS", "true").lower() not in {"1", "true", "yes"}:
        logger.info("Backend corpus auto-ingest disabled by AUTO_INGEST_CORPUS.")
        return
    _ingest_thread_started = True
    Thread(target=_background_corpus_ingest, name="corpus-ingest", daemon=True).start()


def _background_corpus_ingest() -> None:
    try:
        logger.info("Backend corpus auto-ingest started: source_dir=%s", SOURCE_DOCS_DIR)
        from .corpus import ingest_source_documents

        indexed, status = ingest_source_documents()
        logger.info(
            "Backend corpus auto-ingest finished: newly_indexed=%s total_documents=%s chunks=%s",
            len(indexed),
            status.uploaded_documents,
            status.total_chunks,
        )
    except Exception:
        logger.exception("Backend corpus auto-ingest failed.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {exc}"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _ensure_paper_vector_index(paper_id: int) -> None:
    if count_paper_embeddings(paper_id):
        return
    row = get_paper(paper_id)
    if row is None:
        return
    pdf_path = Path(str(row["file_path"]))
    if not pdf_path.exists():
        logger.warning("Skipping vector reindex; missing file: paper_id=%s path=%s", paper_id, pdf_path)
        return
    logger.info("Lazy reindexing paper with current embedding model: paper_id=%s", paper_id)
    _page_count, chunks = extract_pdf_chunks(pdf_path, str(row["name"]))
    from .vector_store import add_chunks

    chunk_count = add_chunks(paper_id, str(row["name"]), chunks)
    update_paper_chunks(paper_id, chunk_count)
    logger.info("Lazy reindex complete: paper_id=%s chunks=%s", paper_id, chunk_count)


def _corpus_status() -> CorpusStatusResponse:
    from .corpus import corpus_status as get_corpus_status

    return get_corpus_status()


async def _store_and_index_upload(file: UploadFile) -> PaperResponse:
    logger.info("Upload request received: filename=%s", file.filename)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"PDF must be smaller than {MAX_UPLOAD_SIZE_MB} MB.")

    safe_name = Path(file.filename or f"paper-{uuid4()}.pdf").name
    stored_path = UPLOAD_DIR / f"{uuid4()}-{safe_name}"
    stored_path.write_bytes(content)

    try:
        page_count, chunks = extract_pdf_chunks(stored_path, safe_name)
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read PDF text: {exc}") from exc

    if not chunks:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No extractable text found in this PDF.")

    paper_id = create_paper(safe_name, stored_path, page_count, 0)
    from .vector_store import add_chunks

    chunk_count = add_chunks(paper_id, safe_name, chunks)
    update_paper_chunks(paper_id, chunk_count)
    logger.info("Upload indexed: paper_id=%s pages=%s chunks=%s", paper_id, page_count, chunk_count)

    paper = get_paper(paper_id)
    return PaperResponse(**dict(paper))


@app.get("/papers", response_model=list[PaperResponse])
def papers() -> list[PaperResponse]:
    return [PaperResponse(**dict(row)) for row in list_papers()]


@app.get("/corpus/status", response_model=CorpusStatusResponse)
def corpus_status() -> CorpusStatusResponse:
    return _corpus_status()


@app.post("/corpus/ingest")
def ingest_corpus() -> dict[str, object]:
    from .corpus import ingest_source_documents

    indexed, status = ingest_source_documents()
    return {"indexed": indexed, "status": status}


@app.get("/papers/{paper_id}/history", response_model=list[ChatMessageResponse])
def paper_history(paper_id: int, session_id: str | None = None) -> list[ChatMessageResponse]:
    if get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return [ChatMessageResponse(**dict(row)) for row in list_chat_history(session_id=session_id, paper_id=paper_id)]


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: int) -> dict[str, str]:
    paper = get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    file_path = Path(str(paper["file_path"]))
    from .vector_store import delete_paper_vectors

    delete_paper_vectors(paper_id)
    delete_paper_record(paper_id)
    file_path.unlink(missing_ok=True)
    logger.info("Deleted paper_id=%s file=%s", paper_id, file_path)
    return {"status": "deleted"}


@app.delete("/papers/{paper_id}/history")
def clear_paper_history(paper_id: int, session_id: str | None = None) -> dict[str, str]:
    if get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    delete_chat_history(paper_id, session_id)
    logger.info("Cleared chat history: paper_id=%s session_id=%s", paper_id, session_id)
    return {"status": "cleared"}


@app.post("/upload", response_model=PaperResponse, status_code=201)
async def upload_paper(file: UploadFile = File(...)) -> PaperResponse:
    return await _store_and_index_upload(file)


@app.post("/upload/batch", response_model=list[PaperResponse], status_code=201)
async def upload_papers(files: list[UploadFile] = File(...)) -> list[PaperResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF.")
    indexed: list[PaperResponse] = []
    for file in files:
        indexed.append(await _store_and_index_upload(file))
    return indexed


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    paper_id = request.paper_id
    if paper_id is not None and get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    if paper_id is None and not list_papers():
        return _empty_corpus_response()
    if paper_id is not None:
        _ensure_paper_vector_index(paper_id)

    from .rag import answer_question

    return answer_question(request.question, paper_id, request.session_id)


def _empty_corpus_response() -> ChatResponse:
    answer = "Information not found in uploaded research papers."
    return ChatResponse(
        answer=answer,
        citations=[],
        simple_explanation=answer,
        key_insights=[],
        confidence=0.0,
        formatted=answer,
        analytics=None,
    )


@app.get("/chat/history", response_model=list[ChatMessageResponse])
def chat_history(session_id: str | None = None, paper_id: int | None = None) -> list[ChatMessageResponse]:
    return [ChatMessageResponse(**dict(row)) for row in list_chat_history(session_id, paper_id)]


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    rows = list_papers()
    if len(rows) < MIN_SOURCE_DOCUMENTS:
        logger.warning("Evaluation running before corpus target is met: uploaded=%s required=%s", len(rows), MIN_SOURCE_DOCUMENTS)

    results: list[EvaluationResult] = []
    for index, item in enumerate(request.questions, start=1):
        if item.expected_paper_id is not None and get_paper(item.expected_paper_id) is None:
            raise HTTPException(status_code=404, detail=f"Expected paper {item.expected_paper_id} not found.")
        from .rag import answer_question

        response = answer_question(item.question, paper_id=None, session_id=f"{request.session_id}-{index}")
        citation_paper_ids = {
            int(row["id"])
            for row in rows
            for citation in response.citations
            if str(row["name"]) == citation.paper_name
        }
        matched_terms = [
            term
            for term in item.expected_answer_terms
            if term.lower() in response.answer.lower()
        ]
        term_score = len(matched_terms) / len(item.expected_answer_terms) if item.expected_answer_terms else response.confidence
        citation_score = 1.0 if response.citations else 0.0
        quality = round((term_score * 0.7) + (citation_score * 0.3), 3)
        retrieval_hit = bool(response.citations)
        if item.expected_paper_id is not None:
            retrieval_hit = item.expected_paper_id in citation_paper_ids
        results.append(
            EvaluationResult(
                question=item.question,
                answer=response.answer,
                retrieval_hit=retrieval_hit,
                answer_quality=quality,
                confidence=response.confidence,
                matched_terms=matched_terms,
                citations=response.citations,
            )
        )

    retrieval_accuracy = round(sum(1 for result in results if result.retrieval_hit) / len(results), 3)
    answer_quality = round(sum(result.answer_quality for result in results) / len(results), 3)
    return EvaluationResponse(
        retrieval_accuracy=retrieval_accuracy,
        answer_quality=answer_quality,
        total_questions=len(results),
        results=results,
    )


@app.get("/papers/{paper_id}/summary", response_model=SummaryResponse)
def paper_summary(paper_id: int) -> SummaryResponse:
    if get_paper(paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    cached = get_cached_summary(paper_id)
    if cached is None:
        logger.info("Summary cache miss for paper_id=%s", paper_id)
        return SummaryResponse(paper_id=paper_id, summary="", citations=[], cached=False)
    logger.info("Summary cache hit for paper_id=%s", paper_id)
    return SummaryResponse(paper_id=paper_id, summary=str(cached["summary"]), citations=[], cached=True)


@app.post("/summarize", response_model=SummaryResponse)
def generate_summary(request: SummaryRequest) -> SummaryResponse:
    logger.info("Summarize request received: paper_id=%s force=%s", request.paper_id, request.force)
    if get_paper(request.paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    try:
        _ensure_paper_vector_index(request.paper_id)
        from .rag import summarize_paper

        summary, citations, cached = summarize_paper(request.paper_id, force=request.force)
        logger.info(
            "Summarize response ready: paper_id=%s cached=%s chars=%s citations=%s",
            request.paper_id,
            cached,
            len(summary),
            len(citations),
        )
        return SummaryResponse(paper_id=request.paper_id, summary=summary, citations=citations, cached=cached)
    except Exception as exc:
        logger.exception("Summary generation failed: paper_id=%s error=%s", request.paper_id, exc)
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {exc}") from exc
