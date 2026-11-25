"""Grounded question answering and prompting-only comparison."""

from __future__ import annotations

from src.llm import complete, resolve_model
from src.prompts import BASIC_QA, CONTEXT_QA, RAG_QA, ROLE_QA, STRUCTURED_QA
from src.retriever import format_context, search

NO_EVIDENCE = "Sufficient supporting evidence was not found in the uploaded research papers."

PROMPT_BUILDERS = {
    "basic": lambda question, context: BASIC_QA.format(question=question),
    "role": lambda question, context: ROLE_QA.format(question=question),
    "structured": lambda question, context: STRUCTURED_QA.format(question=question),
    "context": lambda question, context: CONTEXT_QA.format(question=question, context=context or "(none)"),
    "rag": lambda question, context: RAG_QA.format(question=question, context=context or "(none)"),
}


def ask(question: str, mode: str = "rag", k: int = 5) -> dict:
    mode = mode if mode in PROMPT_BUILDERS else "rag"
    hits = search(question, k=k) if mode in {"context", "rag"} else []
    context = format_context(hits) if hits else ""
    prompt = PROMPT_BUILDERS[mode](question, context)
    result = complete(prompt)
    answer = result["text"]
    grounded = mode in {"context", "rag"}
    unsupported = grounded and (
        not hits or NO_EVIDENCE.lower() in answer.lower() or not context
    )
    return {
        "question": question,
        "mode": mode,
        "model": result["model"],
        "seconds": result["seconds"],
        "answer": answer,
        "sources": [
            {"file_name": h["file_name"], "page": h["page"], "score": round(h["score"], 3), "text": h["text"]}
            for h in hits
        ],
        "unsupported": unsupported,
    }


def run_prompt_modes(question: str) -> list[dict]:
    """Run basic, role, structured, context, and RAG. Only the last two retrieve passages."""
    return [ask(question, mode=mode) for mode in ("basic", "role", "structured", "context", "rag")]


def compare_prompting_and_rag(question: str) -> dict:
    prompting = ask(question, mode="role")
    rag = ask(question, mode="rag")
    return {"question": question, "prompting_only": prompting, "rag": rag, "model": resolve_model()}
