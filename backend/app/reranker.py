from __future__ import annotations

import logging
import os
from functools import lru_cache

from langchain_core.documents import Document

from .config import RERANKER_MODEL, USE_CROSS_ENCODER

logger = logging.getLogger("research_summarizer")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


@lru_cache(maxsize=1)
def _cross_encoder():
    from transformers.utils import logging as transformers_logging
    from sentence_transformers import CrossEncoder

    transformers_logging.disable_progress_bar()
    logger.info("Loading cross-encoder reranker: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


def rerank_documents(question: str, candidates: list[tuple[Document, float]], limit: int) -> list[tuple[Document, float]]:
    if not candidates:
        return []
    if not USE_CROSS_ENCODER:
        query_terms = set(_tokenize(question))
        reranked = []
        for document, score in candidates:
            doc_terms = set(_tokenize(document.page_content))
            lexical = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
            reranked.append((document, float(score + lexical)))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:limit]

    pairs = [(question, document.page_content) for document, _score in candidates]
    scores = _cross_encoder().predict(pairs, show_progress_bar=False)
    reranked = [(document, float(score)) for (document, _old_score), score in zip(candidates, scores)]
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:limit]


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9][a-z0-9+-]{1,}", text.lower())
