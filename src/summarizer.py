"""Structured summaries cached on disk.

A cache entry is reused only when the extracted text, prompt, and model match.
"""

from __future__ import annotations

import hashlib
import json

from src.config import SUMMARIES_DIR, groq_model
from src.llm import complete
from src.pdf_processor import content_sha256, list_papers, load_processed, representative_text
from src.prompts import SUMMARY_PROMPT


def summary_path(file_name: str):
    return SUMMARIES_DIR / f"{file_name.replace('.pdf', '')}.json"


COVERAGE_VERSION = "spread-v2"


def prompt_sha256() -> str:
    return hashlib.sha256(SUMMARY_PROMPT.encode("utf-8")).hexdigest()


def load_summary(file_name: str) -> dict | None:
    path = summary_path(file_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def cache_is_valid(cached: dict | None, record: dict) -> bool:
    if not cached or not cached.get("summary"):
        return False
    return (
        cached.get("content_sha256") == content_sha256(record)
        and cached.get("prompt_sha256") == prompt_sha256()
        and cached.get("model") == groq_model()
        and cached.get("file_name") == record.get("file_name")
        and cached.get("coverage_version") == COVERAGE_VERSION
    )


def summarize_paper(file_name: str, force: bool = False) -> dict:
    record = load_processed(file_name)
    if not record:
        raise RuntimeError(f"No extracted text for {file_name}. Process the PDF first.")
    cached = load_summary(file_name)
    if cache_is_valid(cached, record) and not force:
        cached["status"] = "cached"
        return cached

    prompt = SUMMARY_PROMPT.format(
        file_name=file_name,
        title=record.get("title") or file_name,
        context=representative_text(record),
    )
    result = complete(prompt, temperature=0.1)
    text = (result.get("text") or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty summary. Nothing was saved.")
    payload = {
        "file_name": file_name,
        "title": record.get("title") or file_name,
        "summary": text,
        "model": result["model"],
        "seconds": result["seconds"],
        "content_sha256": content_sha256(record),
        "prompt_sha256": prompt_sha256(),
        "coverage_version": COVERAGE_VERSION,
        "status": "generated",
    }
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path(file_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize_all(force: bool = False) -> list[dict]:
    return [summarize_paper(pdf.name, force=force) for pdf in list_papers()]
