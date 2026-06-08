from __future__ import annotations

import logging
from pathlib import Path

from .config import DOMAIN_NAME, MIN_SOURCE_DOCUMENTS, SOURCE_DOCS_DIR
from .database import create_paper, get_paper, get_paper_by_name, init_db, list_papers, update_paper_chunks
from .models import CorpusStatusResponse, PaperResponse
from .pdf_processing import extract_pdf_chunks
from .vector_store import add_chunks

logger = logging.getLogger("research_summarizer")


def corpus_status() -> CorpusStatusResponse:
    rows = [dict(row) for row in list_papers()]
    uploaded = len(rows)
    return CorpusStatusResponse(
        domain_name=DOMAIN_NAME,
        required_documents=MIN_SOURCE_DOCUMENTS,
        uploaded_documents=uploaded,
        remaining_documents=max(0, MIN_SOURCE_DOCUMENTS - uploaded),
        total_pages=sum(int(row["page_count"]) for row in rows),
        total_chunks=sum(int(row["chunk_count"]) for row in rows),
        ready=uploaded >= MIN_SOURCE_DOCUMENTS,
    )


def ingest_source_documents(source_dir: Path = SOURCE_DOCS_DIR) -> tuple[list[PaperResponse], CorpusStatusResponse]:
    init_db()
    source_dir.mkdir(parents=True, exist_ok=True)
    indexed: list[PaperResponse] = []

    for pdf_path in sorted(source_dir.glob("*.pdf")):
        existing = get_paper_by_name(pdf_path.name)
        if existing is not None and int(existing["chunk_count"]) > 0:
            logger.info("Corpus ingest skipped existing source: %s", pdf_path.name)
            continue

        page_count, chunks = extract_pdf_chunks(pdf_path, pdf_path.name)
        if not chunks:
            logger.warning("Corpus ingest skipped empty PDF text: %s", pdf_path)
            continue

        paper_id = int(existing["id"]) if existing is not None else create_paper(pdf_path.name, pdf_path, page_count, 0)
        chunk_count = add_chunks(paper_id, pdf_path.name, chunks)
        update_paper_chunks(paper_id, chunk_count)
        paper = get_paper(paper_id)
        if paper is not None:
            indexed.append(PaperResponse(**dict(paper)))
        logger.info("Corpus source indexed: paper_id=%s file=%s chunks=%s", paper_id, pdf_path.name, chunk_count)

    return indexed, corpus_status()
