# Backend

## Setup Only

```powershell
cd "C:\AI Research Paper Summarizer"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## Corrective RAG

The backend retrieves citation-aware chunks from the backend-owned source corpus, grades each chunk as relevant, partially relevant, or irrelevant, and runs a corrective retry when the first retrieval is weak. The retry rewrites the query with Groq, retrieves broader hybrid candidates, reranks them, generates a LangChain memory-aware answer, and verifies grounding before returning the response.

Every chat response includes Corrective RAG analytics: retrieval confidence, correction attempts, chunk grades, accepted and rejected chunk counts, final confidence, grounding status, response time, and rewritten query when used.

## Module 6 Corpus Ingestion

Add the 50 domain-specific PDFs here:

```text
data/source_documents
```

Then index them from the backend:

```powershell
cd "C:\AI Research Paper Summarizer"
.\.venv\Scripts\python.exe backend\ingest_corpus.py
```

The script skips already indexed filenames and reports whether the corpus has reached the 50-document requirement.

Useful endpoints:

- `GET /corpus/status`
- `POST /corpus/ingest`
- `POST /chat`
- `POST /evaluate`

## Final Run Command

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Swagger documentation is available at `http://localhost:8000/docs`.
