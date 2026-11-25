"""FAISS index of normalized chunk embeddings (inner product equals cosine similarity)."""

from __future__ import annotations

import json
import pickle

import faiss
import numpy as np

from src.config import VECTOR_DIR
from src.embeddings import chunks_from_paper, embed_texts
from src.pdf_processor import load_all_processed

INDEX_PATH = VECTOR_DIR / "papers.faiss"
META_PATH = VECTOR_DIR / "metadata.pkl"
STAMP_PATH = VECTOR_DIR / "stamp.json"


def _stamp(records: list[dict]) -> dict:
    return {
        "files": [
            {"file_name": r["file_name"], "chars": r.get("char_count"), "pages": r.get("extracted_pages")}
            for r in records
        ]
    }


def index_is_current(records: list[dict] | None = None) -> bool:
    records = records if records is not None else load_all_processed()
    if not INDEX_PATH.exists() or not META_PATH.exists() or not STAMP_PATH.exists():
        return False
    saved = json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    return saved == _stamp(records)


def build_index(force: bool = False) -> dict:
    records = load_all_processed()
    if not records:
        raise RuntimeError("No processed papers found. Extract PDFs first.")
    if index_is_current(records) and not force:
        meta = pickle.loads(META_PATH.read_bytes())
        return {"status": "cached", "chunks": len(meta)}

    rows: list[dict] = []
    for record in records:
        rows.extend(chunks_from_paper(record))
    vectors = embed_texts([row["text"] for row in rows])
    if vectors.ndim != 2 or len(vectors) != len(rows):
        raise RuntimeError("Embedding shape does not match chunk count.")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_bytes(pickle.dumps(rows))
    STAMP_PATH.write_text(json.dumps(_stamp(records)), encoding="utf-8")
    return {"status": "built", "chunks": len(rows), "dimension": int(vectors.shape[1])}


def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError("Vector index is missing. Build it first.")
    index = faiss.read_index(str(INDEX_PATH))
    meta = pickle.loads(META_PATH.read_bytes())
    return index, meta
