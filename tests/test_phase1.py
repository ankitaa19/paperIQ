import unittest

from src.paper_comparator import rows_from_model_text
from src.pdf_processor import load_processed, representative_text
from src.retriever import _diversify, title_bonus
from src.summarizer import COVERAGE_VERSION, cache_is_valid
from src.evaluation import CASES


class Phase1Tests(unittest.TestCase):
    def test_title_bonus_prefers_named_paper(self):
        named = title_bonus(
            "What does Lost in the Middle find about relevant information?",
            {"title": "Lost in the Middle: How Language Models Use Long Contexts", "file_name": "lost_in_the_middle.pdf"},
        )
        other = title_bonus(
            "What does Lost in the Middle find about relevant information?",
            {"title": "SELF-RAG", "file_name": "self_rag.pdf"},
        )
        self.assertGreater(named, other)
        self.assertEqual(other, 0.0)

    def test_diversify_caps_one_file(self):
        ranked = [{"file_name": "a.pdf", "page": i} for i in range(4)]
        ranked += [{"file_name": "b.pdf", "page": 1}]
        chosen = _diversify(ranked, k=3, per_file=2)
        self.assertEqual([row["file_name"] for row in chosen].count("a.pdf"), 2)
        self.assertIn("b.pdf", [row["file_name"] for row in chosen])

    def test_representative_text_includes_middle_page(self):
        record = load_processed("language_models_few_shot.pdf")
        text = representative_text(record, limit=50000)
        middle = record["pages"][len(record["pages"]) // 2]["page"]
        self.assertIn(f"[page {middle}]", text)
        self.assertIn("[page 1]", text)

    def test_old_summary_cache_is_not_reused(self):
        record = {"file_name": "attention_is_all_you_need.pdf", "pages": [{"page": 1, "text": "x"}]}
        cached = {
            "summary": "old",
            "content_sha256": "nope",
            "prompt_sha256": "nope",
            "model": "openai/gpt-oss-20b",
            "file_name": record["file_name"],
        }
        self.assertFalse(cache_is_valid(cached, record))
        self.assertEqual(COVERAGE_VERSION, "spread-v2")

    def test_comparison_table_has_eight_rows(self):
        lines = "\n".join(
            f"{dimension} || A || B"
            for dimension in (
                "Research Objective",
                "Problem Statement",
                "Methodology",
                "Models / Techniques",
                "Experimental Results",
                "Key Findings",
                "Limitations",
                "Conclusion",
            )
        )
        rows = rows_from_model_text(lines)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["Paper A"], "A")

    def test_missing_dimension_is_explicit(self):
        lines = "\n".join(
            f"{dimension} || stated || stated"
            for dimension in (
                "Research Objective",
                "Problem Statement",
                "Methodology",
                "Models / Techniques",
            )
        )
        rows = rows_from_model_text(lines)
        self.assertEqual(rows[0]["Paper A"], "stated")
        self.assertIn("Not clearly specified", rows[4]["Paper A"])
        with self.assertRaises(RuntimeError):
            rows_from_model_text("Research Objective || stated || stated")

    def test_partial_case_is_not_labeled_supported(self):
        kinds = {case["id"]: case["kind"] for case in CASES}
        self.assertEqual(kinds["partial_training_cost"], "partial")
        self.assertEqual(kinds["unsupported"], "unsupported")


if __name__ == "__main__":
    unittest.main()
