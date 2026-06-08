from __future__ import annotations

from app.config import SOURCE_DOCS_DIR
from app.corpus import ingest_source_documents


def main() -> None:
    indexed, status = ingest_source_documents(SOURCE_DOCS_DIR)
    print(f"Source folder: {SOURCE_DOCS_DIR}")
    print(f"New PDFs indexed: {len(indexed)}")
    print(f"Corpus documents: {status.uploaded_documents}/{status.required_documents}")
    print(f"Pages: {status.total_pages}")
    print(f"Chunks: {status.total_chunks}")
    print(f"Ready for Module 6: {'yes' if status.ready else 'no'}")
    if status.remaining_documents:
        print(f"Add {status.remaining_documents} more PDF(s) to meet the requirement.")


if __name__ == "__main__":
    main()
