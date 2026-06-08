from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import SOURCE_DOCS_DIR

TARGET_COUNT = 100
USER_AGENT = "AI-Research-Paper-Summarizer/1.0 (student corpus builder)"
SEARCH_TERMS = [
    "lung cancer treatment immunotherapy open access",
    "colorectal cancer diagnosis treatment open access",
    "prostate cancer treatment biomarkers open access",
    "leukemia cancer therapy diagnosis open access",
    "melanoma immunotherapy cancer open access",
    "pancreatic cancer treatment biomarkers open access",
    "ovarian cancer treatment diagnosis open access",
    "glioma brain cancer treatment imaging open access",
    "cancer radiotherapy chemotherapy immunotherapy open access",
    "machine learning cancer diagnosis medical imaging open access",
]
SSL_CONTEXT = ssl.create_default_context()
try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass


def main() -> None:
    SOURCE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SOURCE_DOCS_DIR / "manifest.json"
    manifest = _load_manifest(manifest_path)
    existing_ids = {item.get("pmcid") for item in manifest}

    for term in SEARCH_TERMS:
        pmc_count = len([item for item in manifest if item.get("source") == "PubMed Central"])
        if pmc_count >= TARGET_COUNT:
            break
        print(f"Searching PMC: {term}")
        pmcids = _search_pmcids(term, retmax=80)
        for pmcid in pmcids:
            pmc_count = len([item for item in manifest if item.get("source") == "PubMed Central"])
            if pmc_count >= TARGET_COUNT:
                break
            if pmcid in existing_ids:
                continue
            record = _oa_record(pmcid)
            if record is None:
                continue
            record["query"] = term
            filename = _filename(record)
            target = SOURCE_DOCS_DIR / filename
            print(f"Downloading {pmc_count + 1:02d}/{TARGET_COUNT}: {record['title']}")
            try:
                _download(record["pdf_url"], target)
            except Exception as exc:
                print(f"Skipped {pmcid}: {exc}")
                target.unlink(missing_ok=True)
                continue
            record["filename"] = filename
            manifest.append(record)
            existing_ids.add(pmcid)
            _save_manifest(manifest_path, manifest)
            time.sleep(0.5)

    _save_manifest(manifest_path, manifest)
    pdf_count = len(list(SOURCE_DOCS_DIR.glob("*.pdf")))
    print(f"Saved PDFs: {pdf_count}")
    if pdf_count < TARGET_COUNT:
        raise SystemExit(f"Only collected {pdf_count} PDFs; rerun later or broaden the search term.")


def _search_pmcids(term: str, retmax: int) -> list[str]:
    params = urllib.parse.urlencode(
        {
            "db": "pmc",
            "term": term,
            "retmax": retmax,
            "sort": "relevance",
            "retmode": "xml",
        }
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    root = _xml(url)
    ids = [node.text or "" for node in root.findall(".//Id")]
    return [f"PMC{value}" for value in ids if value]


def _oa_record(pmcid: str) -> dict[str, str] | None:
    params = urllib.parse.urlencode({"id": pmcid})
    root = _xml(f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?{params}")
    record = root.find(".//record")
    if record is None:
        return None
    title = record.attrib.get("citation", pmcid)
    pdf_url = ""
    for link in record.findall("link"):
        if link.attrib.get("format") == "pdf":
            pdf_url = link.attrib.get("href", "")
            break
    if not pdf_url:
        return None
    if pdf_url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        pdf_url = pdf_url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/", 1)
    pdf_url = pdf_url.replace("/pub/pmc/oa_pdf/", "/pub/pmc/deprecated/oa_pdf/")
    return {
        "source": "PubMed Central",
        "pmcid": pmcid,
        "title": re.sub(r"\s+", " ", title).strip(),
        "pdf_url": pdf_url,
        "source_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
    }


def _xml(url: str) -> ET.Element:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        return ET.fromstring(response.read())


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120, context=SSL_CONTEXT) as response:
        data = response.read()
    if len(data) < 1024 or not data.startswith(b"%PDF"):
        raise ValueError("downloaded file is not a valid PDF")
    target.write_bytes(data)


def _filename(record: dict[str, str]) -> str:
    title = re.sub(r"[^a-zA-Z0-9]+", "-", record["title"]).strip("-").lower()
    title = title[:90] or "paper"
    return f"{record['pmcid']}-{title}.pdf"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
