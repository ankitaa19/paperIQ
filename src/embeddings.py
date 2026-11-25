"""Chunk papers and embed them with all-MiniLM-L6-v2."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(size - overlap, 1)
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunks_from_paper(record: dict) -> list[dict]:
    rows = []
    for page in record.get("pages") or []:
        for piece in chunk_text(page["text"]):
            rows.append(
                {
                    "file_name": record["file_name"],
                    "title": record.get("title") or record["file_name"],
                    "page": page["page"],
                    "text": piece,
                }
            )
    return rows


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    vectors = get_model().encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")
