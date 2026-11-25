"""Side-by-side comparison of two processed papers."""

from __future__ import annotations

import json
import re

from src.llm import complete
from src.pdf_processor import load_processed, representative_text
from src.prompts import COMPARE_PROMPT

DIMENSIONS = [
    "Research Objective",
    "Problem Statement",
    "Methodology",
    "Models / Techniques",
    "Experimental Results",
    "Key Findings",
    "Limitations",
    "Conclusion",
]
MISSING = "Not clearly specified in the available excerpts."
_JSON_FORMAT = {"type": "json_object"}
_ALIASES = {
    "research objective": "Research Objective",
    "objective": "Research Objective",
    "problem statement": "Problem Statement",
    "problem": "Problem Statement",
    "methodology": "Methodology",
    "method": "Methodology",
    "methods": "Methodology",
    "models / techniques": "Models / Techniques",
    "models/techniques": "Models / Techniques",
    "models": "Models / Techniques",
    "techniques": "Models / Techniques",
    "experimental results": "Experimental Results",
    "results": "Experimental Results",
    "key findings": "Key Findings",
    "findings": "Key Findings",
    "limitations": "Limitations",
    "limitation": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
}


def _canonical(name: str) -> str | None:
    cleaned = re.sub(r"[*_`#]+", "", name).strip().strip(":").lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _ALIASES.get(cleaned)


def _cell(value) -> str:
    if value is None:
        return MISSING
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or MISSING


def _pair(value) -> tuple[str, str] | None:
    if isinstance(value, dict):
        left = value.get("paper_a", value.get("Paper A", value.get("a")))
        right = value.get("paper_b", value.get("Paper B", value.get("b")))
        if left is None and right is None:
            return None
        return _cell(left), _cell(right)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _cell(value[0]), _cell(value[1])
    return None


def _from_json(text: str) -> dict[str, tuple[str, str]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    found = {}
    for key, value in payload.items():
        dimension = _canonical(str(key))
        pair = _pair(value)
        if dimension and pair:
            found[dimension] = pair
    return found


def _from_delimiters(text: str) -> dict[str, tuple[str, str]]:
    found = {}
    for line in text.splitlines():
        cleaned = line.strip().strip("|")
        if "||" in cleaned:
            parts = [part.strip(" |*`") for part in cleaned.split("||")]
        elif cleaned.count("|") >= 2:
            parts = [part.strip(" *`") for part in cleaned.split("|")]
        else:
            continue
        parts = [part for part in parts if part and part.lower() not in {"paper a", "paper b", "dimension"}]
        if len(parts) < 3 or set(parts[1:]) <= {"---", ":---", "---:"}:
            continue
        dimension = _canonical(parts[0])
        if dimension:
            found[dimension] = (_cell(parts[1]), _cell(parts[2]))
    return found


def parse_comparison(text: str) -> dict[str, tuple[str, str]]:
    """Read JSON, a markdown table, or || rows. Empty when the text has no fields."""
    if not text or not text.strip():
        return {}
    found = _from_json(text)
    if len(found) >= 4:
        return found
    delimited = _from_delimiters(text)
    if len(delimited) > len(found):
        return delimited
    return found or delimited


def rows_from_fields(found: dict[str, tuple[str, str]]) -> list[dict]:
    if len(found) < 4:
        raise RuntimeError(
            "The comparison did not include enough validated fields, so nothing was saved."
        )
    rows = []
    for dimension in DIMENSIONS:
        paper_a, paper_b = found.get(dimension, (MISSING, MISSING))
        rows.append(
            {
                "Comparison Dimension": dimension,
                "Paper A": paper_a or MISSING,
                "Paper B": paper_b or MISSING,
            }
        )
    return rows


def rows_from_model_text(text: str) -> list[dict]:
    return rows_from_fields(parse_comparison(text))


def compare_papers(file_a: str, file_b: str) -> dict:
    if file_a == file_b:
        raise ValueError("Choose two different papers.")
    left = load_processed(file_a)
    right = load_processed(file_b)
    if not left or not right:
        raise RuntimeError("Both papers must be extracted before they can be compared.")
    prompt = COMPARE_PROMPT.format(
        file_a=file_a,
        file_b=file_b,
        context_a=representative_text(left, limit=7000),
        context_b=representative_text(right, limit=7000),
    )
    calls = 1
    try:
        result = complete(
            prompt,
            temperature=0.0,
            response_format=_JSON_FORMAT,
            retry_status=False,
            max_tokens=1800,
            reasoning_effort="low",
        )
    except RuntimeError:
        result = complete(prompt, temperature=0.0, max_tokens=1800, reasoning_effort="low")
    found = parse_comparison(result["text"])
    seconds = result["seconds"]
    if len(found) < 4:
        calls += 1
        reminder = (
            prompt
            + "\nReply with only the JSON object. Every dimension needs paper_a and paper_b."
        )
        try:
            retry = complete(
                reminder,
                temperature=0.0,
                response_format=_JSON_FORMAT,
                retry_status=False,
                max_tokens=1800,
                reasoning_effort="low",
            )
        except RuntimeError:
            retry = complete(
                reminder,
                temperature=0.0,
                max_tokens=1800,
                reasoning_effort="low",
            )
        seconds += retry["seconds"]
        retry_found = parse_comparison(retry["text"])
        if len(retry_found) >= 4:
            found = retry_found
            result = retry
        else:
            raise RuntimeError(
                "The comparison did not include enough validated fields, so nothing was saved."
            )
    rows = rows_from_fields(found)
    return {
        "file_a": file_a,
        "file_b": file_b,
        "title_a": left.get("title"),
        "title_b": right.get("title"),
        "rows": rows,
        "model": result["model"],
        "seconds": round(seconds, 3),
        "api_calls": calls,
    }
