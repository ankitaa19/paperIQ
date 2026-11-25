"""Paths and environment settings. API keys stay in .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = ROOT / "vector_db"
SUMMARIES_DIR = PROCESSED_DIR / "summaries"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5

load_dotenv(ROOT / ".env")


def groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env (see .env.example).")
    return key


def groq_model() -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
