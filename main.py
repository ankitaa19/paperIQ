"""Command-line entry points for extraction, indexing, and a connectivity check."""

from __future__ import annotations

import argparse
import json

from src.llm import complete, resolve_model
from src.pdf_processor import process_all, verify_expected
from src.vector_store import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperIQ local pipeline")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()

    print(json.dumps(verify_expected(), indent=2))
    if args.check_api:
        model = resolve_model()
        reply = complete("Reply with the single word: ready", model=model)
        print(json.dumps({"model": reply["model"], "reply": reply["text"], "seconds": reply["seconds"]}))
    if args.extract:
        print(json.dumps(process_all(), indent=2))
    if args.index:
        print(json.dumps(build_index(), indent=2))
    if not any([args.check_api, args.extract, args.index]):
        parser.print_help()


if __name__ == "__main__":
    main()
