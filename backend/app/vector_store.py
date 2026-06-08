import logging
import os
import math
import re
from functools import lru_cache
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import (
    CHROMA_DIR,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    ENABLE_CHROMA_WRITE,
    FETCH_K,
    FINAL_CONTEXT_K,
    MIN_RELEVANCE_SCORE,
    MIN_RERANK_SCORE,
    RERANK_TOP_K,
    RERANKER_MODEL,
    TOP_K,
)
from .models import Citation
from .pdf_processing import PaperChunk
from .database import count_paper_embeddings, delete_paper_embeddings, get_paper_embeddings, replace_paper_embeddings

logger = logging.getLogger("research_summarizer")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from transformers.utils import logging as transformers_logging
            from sentence_transformers import SentenceTransformer

            transformers_logging.disable_progress_bar()
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from transformers.utils import logging as transformers_logging
    from sentence_transformers import CrossEncoder

    transformers_logging.disable_progress_bar()
    logger.info("Loading reranker model: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


embedding_function = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name="research_papers_bge_v1_5",
    embedding_function=embedding_function,
    persist_directory=str(CHROMA_DIR),
    collection_metadata={"hnsw:space": "cosine"},
)


def add_chunks(paper_id: int, paper_name: str, chunks: Iterable[PaperChunk]) -> int:
    chunk_list = list(chunks)
    if not chunk_list:
        return 0

    texts = [chunk.text for chunk in chunk_list]
    embeddings = embedding_function.embed_documents(texts)
    replace_paper_embeddings(
        paper_id,
        [
            {
                "chunk_id": chunk.chunk_id,
                "paper_name": paper_name,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "embedding": embedding,
                "is_reference": chunk.is_reference,
            }
            for chunk, embedding in zip(chunk_list, embeddings)
        ],
    )

    ids = [f"{paper_id}:{chunk.chunk_id}" for chunk in chunk_list]
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "paper_id": paper_id,
                "paper_name": paper_name,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "is_reference": bool(chunk.is_reference),
            },
        )
        for chunk in chunk_list
    ]
    if ENABLE_CHROMA_WRITE:
        try:
            vector_store.delete(ids=ids)
        except Exception:
            logger.debug("No existing Chroma vectors to delete for paper_id=%s", paper_id)
        try:
            for index in range(0, len(documents), 64):
                vector_store.add_documents(documents=documents[index : index + 64], ids=ids[index : index + 64])
        except Exception:
            logger.exception("Chroma write failed; SQLite embedding mirror remains available: paper_id=%s", paper_id)
    else:
        logger.info("Chroma write skipped by ENABLE_CHROMA_WRITE=false; SQLite embedding mirror indexed paper_id=%s", paper_id)
    return len(chunk_list)


def count_paper_vectors(paper_id: int) -> int:
    return count_paper_embeddings(paper_id)


def delete_paper_vectors(paper_id: int) -> None:
    rows = get_paper_embeddings(paper_id)
    ids = [f"{paper_id}:{row['chunk_id']}" for row in rows]
    delete_paper_embeddings(paper_id)
    if ids and ENABLE_CHROMA_WRITE:
        try:
            vector_store.delete(ids=ids)
        except Exception:
            logger.exception("Chroma vector delete failed for paper_id=%s", paper_id)


def get_retriever(paper_id: int | None = None, top_k: int = TOP_K):
    search_kwargs: dict[str, object] = {
        "k": top_k,
        "fetch_k": FETCH_K,
        "lambda_mult": 0.5,
    }
    if paper_id is not None:
        search_kwargs["filter"] = {"paper_id": paper_id}
    return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


def retrieve_documents(
    question: str,
    paper_id: int | None = None,
    top_k: int = TOP_K,
    include_references: bool = False,
    search_mode: str = "hybrid",
    use_rerank: bool = True,
    candidate_k: int = FETCH_K,
) -> list[tuple[Document, float]]:
    stored_rows = get_paper_embeddings(paper_id)
    if not stored_rows:
        logger.info("Retrieval rejection: paper_id=%s reason=no_indexed_chunks", paper_id)
        return []
    query_embedding = embedding_function.embed_query(question)
    dense_candidates: list[tuple[Document, float, list[float]]] = []
    for row in stored_rows:
        document = Document(
            page_content=row["text"],
            metadata={
                "paper_id": row["paper_id"],
                "paper_name": row["paper_name"],
                "page_number": row["page_number"],
                "chunk_id": row["chunk_id"],
                "is_reference": row["is_reference"],
            },
        )
        if bool(document.metadata.get("is_reference", False)) and not include_references:
            continue
        embedding = row["embedding"]
        dense_candidates.append((document, _cosine(query_embedding, embedding), list(embedding)))

    if search_mode == "vector":
        dense_candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = [
            (document, score)
            for document, score, _embedding in dense_candidates
            if score >= MIN_RELEVANCE_SCORE
        ][:candidate_k]
    else:
        candidates = _hybrid_rank(question, dense_candidates)[:candidate_k]

    if not candidates:
        logger.info(
            "Retrieval rejection: query=%r paper_id=%s search_mode=%s reason=no_candidates_after_threshold",
            question,
            paper_id,
            search_mode,
        )
        _log_retrieval_debug(question, [], [])
        return []

    if not use_rerank:
        selected = candidates[:top_k]
        _log_retrieval_debug(question, candidates, [(document, score, score) for document, score in selected])
        return selected

    reranked = _rerank(question, candidates)
    filtered = [
        (document, rerank_score, similarity)
        for document, similarity, rerank_score in reranked
        if rerank_score >= MIN_RERANK_SCORE
    ][:FINAL_CONTEXT_K]
    _log_retrieval_debug(question, candidates, filtered)
    return [(document, float(rerank_score)) for document, rerank_score, _similarity in filtered]


def get_page_documents(paper_id: int, page_number: int, include_references: bool = True) -> list[Document]:
    rows = [
        row
        for row in get_paper_embeddings(paper_id)
        if int(row.get("page_number", 0)) == page_number and (include_references or not bool(row.get("is_reference", False)))
    ]
    documents = [
        Document(
            page_content=str(row["text"]),
            metadata={
                "paper_id": int(row["paper_id"]),
                "paper_name": str(row["paper_name"]),
                "page_number": int(row["page_number"]),
                "chunk_id": str(row["chunk_id"]),
                "is_reference": bool(row["is_reference"]),
            },
        )
        for row in sorted(rows, key=lambda item: str(item.get("chunk_id", "")))
    ]
    logger.info(
        "Page metadata retrieval: paper_id=%s page=%s sqlite_chunks=%s",
        paper_id,
        page_number,
        len(documents),
    )
    if documents:
        return documents

    logger.info("Page metadata retrieval retrying Chroma: paper_id=%s page=%s", paper_id, page_number)
    return _get_page_documents_from_chroma(paper_id, page_number, include_references)


def get_paper_contexts(paper_id: int) -> tuple[list[str], list[Citation]]:
    rows = sorted(
        get_paper_embeddings(paper_id),
        key=lambda row: (int(row.get("page_number", 0)), str(row.get("chunk_id", ""))),
    )

    contexts: list[str] = []
    citations: list[Citation] = []
    for row in rows:
        contexts.append(str(row["text"]))
        citations.append(
            Citation(
                paper_name=str(row["paper_name"]),
                page_number=int(row["page_number"]),
                chunk_id=str(row["chunk_id"]),
                score=1.0,
            )
        )
    return contexts, citations


def _get_page_documents_from_chroma(paper_id: int, page_number: int, include_references: bool) -> list[Document]:
    filters = [
        {"$and": [{"paper_id": {"$eq": paper_id}}, {"page_number": {"$eq": page_number}}]},
        {"paper_id": paper_id, "page_number": page_number},
    ]
    for where in filters:
        try:
            result = vector_store.get(where=where, include=["documents", "metadatas"])
        except Exception:
            logger.exception("Chroma page metadata query failed: paper_id=%s page=%s filter=%s", paper_id, page_number, where)
            continue

        texts = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        documents: list[Document] = []
        for text, metadata in zip(texts, metadatas):
            metadata = dict(metadata or {})
            if bool(metadata.get("is_reference", False)) and not include_references:
                continue
            documents.append(Document(page_content=str(text), metadata=metadata))
        if documents:
            logger.info("Chroma page metadata query found chunks: paper_id=%s page=%s chunks=%s", paper_id, page_number, len(documents))
            return sorted(documents, key=lambda document: str(document.metadata.get("chunk_id", "")))
    logger.info("Chroma page metadata query found no chunks: paper_id=%s page=%s", paper_id, page_number)
    return []


def search_context(question: str, paper_id: int | None = None, top_k: int = TOP_K) -> tuple[list[str], list[Citation], float]:
    documents_and_scores = retrieve_documents(question, paper_id, top_k)

    contexts: list[str] = []
    citations: list[Citation] = []
    scores: list[float] = []

    for document, score in documents_and_scores:
        metadata = document.metadata
        scores.append(score)
        contexts.append(document.page_content)
        citations.append(
            Citation(
                paper_name=str(metadata["paper_name"]),
                page_number=int(metadata["page_number"]),
                chunk_id=str(metadata["chunk_id"]),
                score=round(score, 3),
            )
        )

    confidence = round(sum(scores) / len(scores), 3) if scores else 0.0
    return contexts, citations, confidence


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def _hybrid_rank(
    question: str,
    dense_candidates: list[tuple[Document, float, list[float]]],
) -> list[tuple[Document, float]]:
    if not dense_candidates:
        return []

    documents = [document for document, _dense, _embedding in dense_candidates]
    dense_scores = [dense for _document, dense, _embedding in dense_candidates]
    bm25_scores = _bm25_scores(question, [document.page_content for document in documents])
    normalized_dense = _min_max(dense_scores)
    normalized_bm25 = _min_max(bm25_scores)

    ranked: list[tuple[Document, float]] = []
    for document, dense, bm25, dense_norm, bm25_norm in zip(
        documents,
        dense_scores,
        bm25_scores,
        normalized_dense,
        normalized_bm25,
    ):
        hybrid_score = (DENSE_WEIGHT * dense_norm) + (BM25_WEIGHT * bm25_norm)
        if hybrid_score < MIN_RELEVANCE_SCORE:
            continue
        document.metadata["dense_score"] = round(dense, 4)
        document.metadata["bm25_score"] = round(bm25, 4)
        document.metadata["hybrid_score"] = round(hybrid_score, 4)
        ranked.append((document, hybrid_score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    tokenized_docs = [_tokenize(document) for document in documents]
    query_terms = _tokenize(query)
    if not tokenized_docs or not query_terms:
        return [0.0 for _document in documents]

    doc_count = len(tokenized_docs)
    avg_doc_len = sum(len(document) for document in tokenized_docs) / max(doc_count, 1)
    document_frequencies: dict[str, int] = {}
    for document in tokenized_docs:
        for term in set(document):
            document_frequencies[term] = document_frequencies.get(term, 0) + 1

    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for document in tokenized_docs:
        term_counts: dict[str, int] = {}
        for term in document:
            term_counts[term] = term_counts.get(term, 0) + 1
        score = 0.0
        doc_len = len(document) or 1
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequencies.get(term, 0)
            idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            denominator = frequency + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * ((frequency * (k1 + 1)) / denominator)
        scores.append(score)
    return scores


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9+-]{1,}", text.lower())


def _min_max(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(high, low):
        return [1.0 if high > 0 else 0.0 for _score in scores]
    return [(score - low) / (high - low) for score in scores]


def _rerank(question: str, candidates: list[tuple[Document, float]]) -> list[tuple[Document, float, float]]:
    pairs = [(question, document.page_content) for document, _similarity in candidates]
    scores = _get_cross_encoder().predict(pairs, show_progress_bar=False)
    reranked = [
        (document, similarity, float(score))
        for (document, similarity), score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda item: item[2], reverse=True)
    return reranked[:RERANK_TOP_K]


def _log_retrieval_debug(
    question: str,
    candidates: list[tuple[Document, float]],
    filtered: list[tuple[Document, float, float]],
) -> None:
    logger.info("Retrieval debug: rewritten_query=%r candidates=%s final=%s", question, len(candidates), len(filtered))
    filtered_lookup = {str(document.metadata.get("chunk_id", "")): rerank for document, rerank, _sim in filtered}
    for document, similarity in candidates:
        metadata = document.metadata
        chunk_id = str(metadata.get("chunk_id", ""))
        preview = " ".join(document.page_content.split())[:150]
        logger.info(
            "Retrieved chunk: chunk_id=%s page=%s dense=%s bm25=%s hybrid=%.3f reranker=%s reference=%s preview=%r",
            chunk_id,
            metadata.get("page_number"),
            metadata.get("dense_score"),
            metadata.get("bm25_score"),
            similarity,
            round(filtered_lookup[chunk_id], 3) if chunk_id in filtered_lookup else None,
            bool(metadata.get("is_reference", False)),
            preview,
        )
