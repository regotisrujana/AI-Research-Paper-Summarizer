from pathlib import Path

from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = BASE_DIR / "backend"
ENV_PATH = BACKEND_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
PACKAGED_SOURCE_DOCS_DIR = BACKEND_DIR / "source_documents"
SOURCE_DOCS_DIR = Path(os.getenv("SOURCE_DOCS_DIR", PACKAGED_SOURCE_DOCS_DIR if PACKAGED_SOURCE_DOCS_DIR.exists() else DATA_DIR / "source_documents"))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_semantic"
DB_PATH = DATA_DIR / "app.sqlite3"

load_dotenv(ENV_PATH)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
TOP_K = 6
FETCH_K = 20
RERANK_TOP_K = 6
FINAL_CONTEXT_K = 5
MMR_LAMBDA_MULT = 0.5
MIN_RELEVANCE_SCORE = 0.05
MIN_RERANK_SCORE = -1.5
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
ENABLE_CHROMA_WRITE = os.getenv("ENABLE_CHROMA_WRITE", "false").lower() == "true"
DOMAIN_NAME = os.getenv("DOMAIN_NAME", "Cancer research corpus")
MIN_SOURCE_DOCUMENTS = int(os.getenv("MIN_SOURCE_DOCUMENTS", "50"))

ALLOWED_EXTENSIONS = {".pdf"}
MAX_UPLOAD_SIZE_MB = 30
