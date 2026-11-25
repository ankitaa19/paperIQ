"""Extract page-level text from research PDFs with PyMuPDF."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from src.config import PAPERS_DIR, PROCESSED_DIR, SUMMARIES_DIR

EXPECTED_PAPERS = [
    "attention_is_all_you_need.pdf",
    "retrieval_augmented_generation.pdf",
    "language_models_few_shot.pdf",
    "lost_in_the_middle.pdf",
    "self_rag.pdf",
]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> dict:
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {path.name}")
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not open {path.name}: {exc}") from exc

    if document.is_encrypted and not document.authenticate(""):
        document.close()
        raise ValueError(f"{path.name} is encrypted and cannot be read.")

    meta = document.metadata or {}
    pages = []
    for index, page in enumerate(document, start=1):
        text = clean_text(page.get_text("text") or "")
        if text:
            pages.append({"page": index, "text": text})
    page_count = document.page_count
    document.close()

    if not pages:
        raise ValueError(f"{path.name} has no extractable text.")

    title = (meta.get("title") or "").strip()
    if not title or title.lower() in {"untitled", "microsoft word"}:
        title = _guess_title(pages[0]["text"], path.stem)

    return {
        "file_name": path.name,
        "stem": path.stem,
        "title": title,
        "author": (meta.get("author") or "").strip(),
        "page_count": page_count,
        "extracted_pages": len(pages),
        "char_count": sum(len(p["text"]) for p in pages),
        "pages": pages,
    }


_TITLE_SKIP = (
    "arxiv",
    "permission",
    "copyright",
    "abstract",
    "provided proper",
    "google hereby",
    "reproduce the",
    "proceedings",
    "preprint",
    "@",
    "university",
    "google brain",
    "google research",
)


def _guess_title(first_page: str, fallback: str) -> str:
    chosen = []
    for line in first_page.splitlines():
        line = line.strip(" *†")
        lower = line.lower()
        if not (12 <= len(line) <= 120):
            continue
        if any(token in lower for token in _TITLE_SKIP):
            continue
        words = line.split()
        if len(words) < 3 or len(words) > 16:
            continue
        if sum(ch.isalpha() for ch in line) < 10:
            continue
        chosen.append(line)
        if len(chosen) == 2:
            break
    if not chosen:
        return fallback.replace("_", " ")
    title = chosen[0]
    if title.lower().endswith((" for", " of", " and", " to", " the", ":")) and len(chosen) > 1:
        title = f"{title} {chosen[1]}"
    return title[:180]


def processed_path(file_name: str) -> Path:
    return PROCESSED_DIR / f"{Path(file_name).stem}.json"


def save_processed(record: dict) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = processed_path(record["file_name"])
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_processed(file_name: str) -> dict | None:
    path = processed_path(file_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_papers() -> list[Path]:
    if not PAPERS_DIR.exists():
        return []
    return sorted(p for p in PAPERS_DIR.glob("*.pdf") if p.is_file())


def load_all_processed() -> list[dict]:
    records = []
    for pdf in list_papers():
        record = load_processed(pdf.name)
        if record:
            records.append(record)
    return records


def process_all(force: bool = False) -> list[dict]:
    summaries = []
    for pdf in list_papers():
        cached = load_processed(pdf.name)
        if cached and not force:
            summaries.append(
                {
                    "file_name": pdf.name,
                    "title": cached.get("title"),
                    "pages": cached.get("extracted_pages"),
                    "chars": cached.get("char_count"),
                    "status": "cached",
                }
            )
            continue
        record = extract_pdf(pdf)
        save_processed(record)
        summaries.append(
            {
                "file_name": pdf.name,
                "title": record["title"],
                "pages": record["extracted_pages"],
                "chars": record["char_count"],
                "status": "extracted",
            }
        )
    return summaries


def is_sample(file_name: str) -> bool:
    return Path(file_name).name in EXPECTED_PAPERS


def content_sha256(record: dict) -> str:
    payload = "\n".join(page["text"] for page in record.get("pages") or [])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def add_uploaded_pdf(name: str, data: bytes) -> dict:
    """Save a user PDF, extract it, and reject duplicates or unreadable files."""
    safe = Path(name).name
    if safe != name or not safe.lower().endswith(".pdf"):
        raise ValueError("Choose a PDF file name without a folder path.")
    if not data:
        raise ValueError(f"{safe} is empty.")
    dest = PAPERS_DIR / safe
    if dest.exists():
        raise FileExistsError(f"{safe} is already in the library and was left unchanged.")
    digest = hashlib.sha256(data).hexdigest()
    for existing in list_papers():
        if hashlib.sha256(existing.read_bytes()).hexdigest() == digest:
            raise FileExistsError(
                f"{safe} has the same contents as {existing.name}, which is already in the library."
            )
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    try:
        record = extract_pdf(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    record["source"] = "sample" if is_sample(safe) else "uploaded"
    record["file_sha256"] = digest
    record["content_sha256"] = content_sha256(record)
    save_processed(record)
    return record


def remove_uploaded_paper(file_name: str) -> None:
    """Delete one uploaded PDF and its processed files. Sample papers stay."""
    safe = Path(file_name).name
    if is_sample(safe):
        raise ValueError("The five sample papers cannot be removed.")
    pdf = PAPERS_DIR / safe
    if not pdf.exists():
        raise FileNotFoundError(f"{safe} is not in the library.")
    pdf.unlink()
    processed_path(safe).unlink(missing_ok=True)
    (SUMMARIES_DIR / f"{Path(safe).stem}.json").unlink(missing_ok=True)


def verify_expected() -> dict[str, bool]:
    present = {p.name for p in list_papers()}
    return {name: name in present for name in EXPECTED_PAPERS}


_SECTION_HINTS = ("limitation", "experiment", "result", "finding", "method", "conclusion")


def representative_text(record: dict, limit: int = 14000) -> str:
    """Sample the start, middle, and end, plus one middle page that names results or limitations."""
    pages = record.get("pages") or []
    if not pages:
        return ""
    count = len(pages)
    indexes = {0, count // 4, count // 2, (3 * count) // 4, count - 1}
    middle = range(max(count // 5, 1), max(count - 1, 1))
    for index in middle:
        lowered = pages[index]["text"].lower()
        if any(hint in lowered for hint in _SECTION_HINTS):
            indexes.add(index)
            break
    parts = []
    used = 0
    for index in sorted(indexes):
        page = pages[index]
        block = f"[page {page['page']}]\n{page['text']}"
        if used + len(block) > limit:
            remaining = limit - used
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)
