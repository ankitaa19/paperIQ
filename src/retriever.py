"""Semantic retrieval over the FAISS paper index.

FAISS still supplies the similarity scores. A small title overlap bonus and a
per-paper cap stop one paper's near-duplicate chunks from crowding out the paper
named in the question.
"""

from __future__ import annotations

import re

from src.config import TOP_K
from src.embeddings import embed_texts
from src.vector_store import load_index

_STOP = {
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "for", "from",
    "what", "which", "where", "when", "who", "why", "how", "does", "do",
    "is", "are", "was", "were", "about", "with", "that", "this", "these",
    "paper", "papers", "find", "found", "using",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 3 and token not in _STOP}


def title_bonus(query: str, hit: dict) -> float:
    """Boost a chunk when the question uses words from that paper's title or file name."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    label = f"{hit.get('title', '')} {hit.get('file_name', '').replace('_', ' ')}"
    label_tokens = _tokens(label)
    overlap = query_tokens & label_tokens
    if not overlap:
        return 0.0
    return min(0.12 * len(overlap), 0.36)


def _diversify(ranked: list[dict], k: int, per_file: int = 2) -> list[dict]:
    chosen: list[dict] = []
    counts: dict[str, int] = {}
    overflow: list[dict] = []
    for hit in ranked:
        name = hit["file_name"]
        if counts.get(name, 0) >= per_file:
            overflow.append(hit)
            continue
        chosen.append(hit)
        counts[name] = counts.get(name, 0) + 1
        if len(chosen) == k:
            return chosen
    for hit in overflow:
        chosen.append(hit)
        if len(chosen) == k:
            break
    return chosen


def search(query: str, k: int = TOP_K) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    index, meta = load_index()
    pool = min(index.ntotal, max(k * 8, 24))
    vector = embed_texts([query])
    scores, ids = index.search(vector, pool)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = dict(meta[int(idx)])
        row["vector_score"] = float(score)
        overlap = len(_tokens(query) & _tokens(row.get("text", "")))
        row["score"] = float(score) + title_bonus(query, row) + min(0.035 * overlap, 0.21)
        hits.append(row)
    hits.sort(key=lambda row: row["score"], reverse=True)
    return _diversify(hits, min(k, len(hits)))


def format_context(hits: list[dict], max_chars: int = 7000) -> str:
    blocks = []
    used = 0
    for hit in hits:
        block = (
            f"Source: {hit['file_name']} | page {hit['page']} | score {hit['score']:.3f}\n"
            f"{hit['text']}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
