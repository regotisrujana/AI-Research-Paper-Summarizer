from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass(frozen=True)
class PaperChunk:
    text: str
    page_number: int
    chunk_id: str
    is_reference: bool = False


def extract_pdf_chunks(pdf_path: Path, paper_name: str) -> tuple[int, list[PaperChunk]]:
    reader = PdfReader(str(pdf_path))
    chunks: list[PaperChunk] = []
    in_references = False

    for page_index, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if not text:
            continue
        page_is_reference = in_references or _starts_reference_section(text)
        if page_is_reference:
            in_references = True
        chunks.extend(_chunk_text(text, paper_name, page_index, page_is_reference))

    return len(reader.pages), chunks


def _chunk_text(text: str, paper_name: str, page_number: int, is_reference: bool = False) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    start = 0
    local_index = 1
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                PaperChunk(
                    text=chunk_text,
                    page_number=page_number,
                    chunk_id=f"{paper_name}:p{page_number}:c{local_index}",
                    is_reference=is_reference or _looks_like_reference_chunk(chunk_text),
                )
            )
        start += step
        local_index += 1

    return chunks


def _starts_reference_section(text: str) -> bool:
    prefix = text[:220].lower()
    return bool(prefix.startswith("references") or prefix.startswith("bibliography") or " references " in prefix[:80])


def _looks_like_reference_chunk(text: str) -> bool:
    lower = text.lower()
    if _starts_reference_section(text):
        return True
    year_markers = lower.count(" doi ") + lower.count(" et al") + lower.count("http")
    numbered_refs = sum(1 for token in lower.split()[:80] if token.rstrip(".").isdigit())
    return year_markers >= 3 or numbered_refs >= 8
