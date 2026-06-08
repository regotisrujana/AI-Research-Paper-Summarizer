# Cancer Research RAG Chatbot with Retrieval Accuracy and Answer Quality Evaluation

A local React + FastAPI dashboard for a backend-owned, domain-specific RAG corpus. The backend indexes at least 50 source PDFs, retrieves citation-grounded evidence, answers questions with Corrective RAG, and reports retrieval accuracy plus answer quality for Module 6 evaluation.

## Current Stack

- Frontend: React, TailwindCSS, Vite
- Backend: FastAPI, Pydantic, SQLite
- RAG: LangChain, ChromaDB integration, BM25 + dense retrieval, CrossEncoder reranking
- LLM: Groq Llama through `langchain-groq`
- Memory: `ConversationBufferMemory` restored from SQLite chat history
- Storage: backend source PDFs on disk, paper metadata, cached summaries, evaluation runs, and chat history in SQLite

## Module 6 Requirement

Build a domain-specific RAG chatbot with at least 50 source documents; evaluate retrieval accuracy and answer quality.

This project now uses a cancer research corpus collected from open-access PubMed Central PDFs. The corpus includes multiple cancer topics, including breast cancer, lung cancer, colorectal cancer, prostate cancer, leukemia, melanoma, pancreatic cancer, ovarian cancer, glioma, diagnosis, imaging, biomarkers, chemotherapy, radiotherapy, and immunotherapy. It supports the requirement as follows:

- Put the domain PDFs in `data/source_documents`.
- Keep at least 50 PDFs in that folder. The current backend corpus contains 100 PDFs.
- Run backend ingestion so the backend, not the user, indexes the source documents.
- Ask questions against all corpus documents by leaving the paper selection on `All corpus documents`.
- Run the evaluation panel or call `POST /evaluate` with test questions and expected answer terms.

Backend corpus ingestion:

```powershell
cd "C:\AI Research Paper Summarizer"
.\.venv\Scripts\python.exe backend\ingest_corpus.py
```

Corpus status:

```text
GET /corpus/status
```

## Corrective RAG Flow

```text
User query
-> hybrid retrieval over backend-indexed corpus chunks
-> retrieval grading: relevant, partially relevant, or irrelevant
-> corrective retry with Groq query rewriting when evidence is weak
-> broader hybrid retrieval and CrossEncoder reranking
-> LangChain ConversationalRetrievalChain with memory
-> Groq citation-grounded answer
-> grounding verification and unsupported-claim removal
-> final answer with citations and Corrective RAG analytics
```

## Corrective Behavior

- Greetings and thanks are answered conversationally without retrieval.
- Corpus questions retrieve only from backend-indexed source PDFs.
- Weak retrieval triggers one corrective loop with a rewritten academic search query, broader hybrid retrieval, and reranking.
- Each retrieved chunk is graded as `relevant`, `partially relevant`, or `irrelevant`.
- The final answer is checked against accepted chunks before it is returned.
- Unsupported lines are removed before the response is shown.
- Empty sections are removed completely.

## Not Found Rule

If no supported evidence remains after correction and grounding verification, the answer is:

```text
Information not found in uploaded research papers.
```

## Prompts

Prompt templates are separated by task:

- factual QA
- beginner explanation
- summary
- comparison
- research gaps
- viva questions
- follow-up questions

See `backend/app/prompts.py`.

## Analytics

Each chat response includes:

- first retrieval confidence
- correction attempts
- final confidence
- chunk grades
- accepted and rejected chunk counts
- grounding check result
- response time
- rewritten query, when used

The React dashboard shows these in the Corrective RAG Details panel.

## API Endpoints

- `GET /corpus/status`
- `POST /corpus/ingest`
- `GET /papers`
- `POST /upload`
- `POST /upload/batch`
- `POST /chat`
- `POST /evaluate`
- `GET /chat/history`
- `GET /papers/{paper_id}/history`
- `DELETE /papers/{paper_id}`
- `DELETE /papers/{paper_id}/history`
- `GET /papers/{paper_id}/summary`
- `POST /summarize`

Deleting a paper removes:

- SQLite paper metadata
- related chat history
- cached summary
- embedding mirror rows
- ChromaDB chunks when Chroma writes are enabled
- stored local PDF file

## Environment

Create `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-8b-instant
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-research-paper-summarizer
LANGCHAIN_API_KEY=your_langchain_key_optional
DOMAIN_NAME=Cancer research corpus
MIN_SOURCE_DOCUMENTS=50
```

Optional:

```env
ENABLE_CHROMA_WRITE=true
```

The app keeps a stable SQLite embedding mirror for local retrieval. ChromaDB integration is still present and can be enabled with `ENABLE_CHROMA_WRITE=true`.

## Setup

Backend dependencies:

```powershell
cd "C:\AI Research Paper Summarizer"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Frontend dependencies:

```powershell
cd "C:\AI Research Paper Summarizer\frontend"
npm install
```

## Sample Queries

- `Hi`
- `What are the common objectives across this cancer research corpus?`
- `What are the key methods used in this domain?`
- `Explain cancer treatment approaches simply.`
- `Summarize this paper.` when one paper is selected
- `What limitations are reported across the documents?`
- `Compare findings across the corpus.`
- `What research gaps are mentioned?`
- `Create viva questions from this domain.`

## Final Run Commands

Backend:

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```powershell
npm run dev
```
