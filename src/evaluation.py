"""Live checks for retrieval, refusal, latency, and human-reviewable answers.

Retrieval and the unsupported-question refusal are deterministic.
Correctness of a supported answer is not reduced to keyword overlap and is not
reported as a percentage. The reference answer is shown beside the model answer
so a person can judge it.
"""

from __future__ import annotations

import time

from src.rag_pipeline import NO_EVIDENCE, ask
from src.retriever import search

CASES = [
    {
        "id": "transformer",
        "kind": "supported",
        "question": "What attention mechanism does the Transformer use instead of recurrence?",
        "expect_file": "attention_is_all_you_need.pdf",
        "reference": (
            "The Transformer uses attention, including scaled dot-product and multi-head "
            "attention, instead of recurrence."
        ),
    },
    {
        "id": "lost_middle",
        "kind": "supported",
        "question": "What does Lost in the Middle find about where relevant information is placed?",
        "expect_file": "lost_in_the_middle.pdf",
        "reference": (
            "Performance is often strongest when the relevant information is at the beginning "
            "or the end of the context, and weaker when it is in the middle."
        ),
    },
    {
        "id": "partial_training_cost",
        "kind": "partial",
        "question": (
            "What GPU-hours did Lost in the Middle spend to train its own language model, "
            "and what position effect does the paper report?"
        ),
        "expect_file": "lost_in_the_middle.pdf",
        "reference": (
            "The position effect is reported. A GPU-hour training cost for a model trained "
            "by that paper is not a finding the excerpts are expected to establish. A partial "
            "answer should state the position effect and say the training-cost part is not "
            "clearly specified."
        ),
    },
    {
        "id": "unsupported",
        "kind": "unsupported",
        "question": "What was the 2024 crop yield in Maharashtra according to these papers?",
        "expect_file": None,
        "reference": NO_EVIDENCE,
    },
]


def retrieval_check(question: str, expect_file: str | None, k: int = 5) -> dict:
    started = time.perf_counter()
    hits = search(question, k=k)
    top_files = []
    for hit in hits:
        if hit["file_name"] not in top_files:
            top_files.append(hit["file_name"])
    found = None if expect_file is None else expect_file in top_files
    return {
        "seconds": round(time.perf_counter() - started, 3),
        "top_files": top_files,
        "top_score": round(hits[0]["score"], 3) if hits else None,
        "expected_file_retrieved": found,
        "hits": [
            {"file_name": h["file_name"], "page": h["page"], "score": round(h["score"], 3)}
            for h in hits
        ],
    }


def deterministic_checks(case: dict, retrieval: dict, rag: dict) -> dict:
    """Checks a person can recompute. These are not a semantic accuracy percentage."""
    cited = {source["file_name"] for source in rag["sources"]}
    kind = case["kind"]
    if kind == "unsupported":
        refusal = NO_EVIDENCE.lower() in rag["answer"].lower()
        return {
            "retrieval_rule": "no expected paper",
            "retrieval_pass": retrieval["expected_file_retrieved"] is None,
            "refusal_pass": refusal,
            "cited_expected_file": None,
            "correctness": "not scored; refusal check above is the deterministic result",
        }
    cited_expected = case["expect_file"] in cited
    return {
        "retrieval_rule": "expected paper appears in the top results",
        "retrieval_pass": retrieval["expected_file_retrieved"] is True,
        "refusal_pass": None,
        "cited_expected_file": cited_expected,
        "correctness": "human review against the reference answer; no automatic percentage",
    }


def run_evaluation() -> dict:
    rows = []
    for case in CASES:
        retrieval = retrieval_check(case["question"], case["expect_file"])
        prompting = ask(case["question"], mode="role")
        rag = ask(case["question"], mode="rag")
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "question": case["question"],
                "reference": case["reference"],
                "retrieval": retrieval,
                "checks": deterministic_checks(case, retrieval, rag),
                "prompting_only": {
                    "seconds": prompting["seconds"],
                    "answer": prompting["answer"],
                    "model": prompting["model"],
                    "sources": prompting["sources"],
                },
                "rag": {
                    "seconds": rag["seconds"],
                    "answer": rag["answer"],
                    "unsupported": rag["unsupported"],
                    "sources": [
                        {"file_name": s["file_name"], "page": s["page"]} for s in rag["sources"]
                    ],
                    "model": rag["model"],
                },
            }
        )
    return {"cases": rows}
