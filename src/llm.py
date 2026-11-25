"""Groq chat helper. The API key is read from the environment and never logged."""

from __future__ import annotations

import time

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from src.config import groq_api_key, groq_model


def client() -> Groq:
    return Groq(api_key=groq_api_key())


def list_model_ids() -> list[str]:
    models = client().models.list()
    return sorted(m.id for m in models.data)


CHAT_SKIP = ("prompt-guard", "whisper", "orpheus", "safeguard")
CHAT_PREFERENCE = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "allam-2-7b",
)


def _chat_models(available: list[str]) -> list[str]:
    return [m for m in available if not any(skip in m.lower() for skip in CHAT_SKIP)]


def resolve_model(preferred: str | None = None) -> str:
    preferred = preferred or groq_model()
    available = list_model_ids()
    chat = _chat_models(available)
    if preferred in chat:
        return preferred
    for candidate in CHAT_PREFERENCE:
        if candidate in chat:
            return candidate
    if chat:
        return chat[0]
    raise RuntimeError("No chat models are available on this Groq account.")


def complete(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.1,
    response_format: dict | None = None,
    retry_status: bool = True,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    model_id = model or resolve_model()
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(3):
        try:
            kwargs = {
                "model": model_id,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client().chat.completions.create(**kwargs)
            message = response.choices[0].message
            text = (message.content or "").strip()
            if not text:
                text = (
                    getattr(message, "reasoning", None)
                    or getattr(message, "reasoning_content", None)
                    or ""
                ).strip()
            return {
                "text": text,
                "model": model_id,
                "seconds": round(time.perf_counter() - started, 3),
            }
        except RateLimitError as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
        except APIStatusError as exc:
            last_error = exc
            if not retry_status:
                break
            time.sleep(1.5 * (attempt + 1))
        except APIConnectionError as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Groq request failed after retries: {last_error}")
