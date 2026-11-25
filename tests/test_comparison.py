"""Parser checks for comparison output that is not eight || lines."""

import unittest

from src.paper_comparator import MISSING, parse_comparison, rows_from_model_text

VALID_JSON = """```json
{
  "Research Objective": {"paper_a": "A objective", "paper_b": "B objective"},
  "Problem Statement": {"paper_a": "A problem", "paper_b": "B problem"},
  "Methodology": {"paper_a": "A method", "paper_b": "B method"},
  "Models / Techniques": {"paper_a": "A model", "paper_b": "B model"},
  "Experimental Results": {"paper_a": "A result", "paper_b": "B result"},
  "Key Findings": {"paper_a": "A finding", "paper_b": "B finding"},
  "Limitations": {"paper_a": "", "paper_b": "B limit"},
  "Conclusion": {"paper_a": "A close", "paper_b": "B close"}
}
```"""

MARKDOWN = """
| Dimension | Paper A | Paper B |
| --- | --- | --- |
| Research Objective | reduce sequence cost | retrieve then generate |
| Problem Statement | recurrence is slow | parametric memory is stale |
| Methodology | encoder decoder attention | dense retrieval plus generator |
| Models / Techniques | Transformer | DPR and BART |
| Experimental Results | BLEU 28.4 | improves open QA |
| Key Findings | attention is sufficient | retrieval helps knowledge tasks |
| Limitations | not stated | index can be outdated |
| Conclusion | new architecture | generation uses retrieved text |
"""


class ComparisonParseTests(unittest.TestCase):
    def test_json_fence_builds_eight_rows(self):
        rows = rows_from_model_text(VALID_JSON)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["Paper A"], "A objective")
        self.assertEqual(rows[0]["Paper B"], "B objective")
        self.assertEqual(rows[6]["Paper A"], MISSING)
        self.assertEqual(rows[6]["Paper B"], "B limit")

    def test_markdown_table(self):
        found = parse_comparison(MARKDOWN)
        rows = rows_from_model_text(MARKDOWN)
        self.assertEqual(len(found), 8)
        self.assertEqual(rows[3]["Paper A"], "Transformer")
        self.assertEqual(rows[3]["Paper B"], "DPR and BART")

    def test_missing_fields_are_explicit(self):
        partial = {
            "Research Objective": {"paper_a": "stated", "paper_b": "stated"},
            "Problem Statement": {"paper_a": "stated", "paper_b": "stated"},
            "Methodology": {"paper_a": "stated", "paper_b": "stated"},
            "Key Findings": {"paper_a": "stated", "paper_b": "stated"},
        }
        import json

        rows = rows_from_model_text(json.dumps(partial))
        self.assertEqual(rows[4]["Paper A"], MISSING)
        self.assertEqual(rows[7]["Paper B"], MISSING)

    def test_truncated_json_is_rejected(self):
        truncated = '{"Research Objective": {"paper_a": "A", "paper_b": "B"}, "Problem'
        with self.assertRaises(RuntimeError):
            rows_from_model_text(truncated)

    def test_prose_without_fields_is_rejected(self):
        with self.assertRaises(RuntimeError):
            rows_from_model_text("Both papers study language models and the results look strong.")

    def test_pipe_rows_still_parse(self):
        lines = "\n".join(f"{name} || left || right" for name in (
            "Research Objective",
            "Problem Statement",
            "Methodology",
            "Models / Techniques",
            "Experimental Results",
            "Key Findings",
            "Limitations",
            "Conclusion",
        ))
        rows = rows_from_model_text(lines)
        self.assertEqual(rows[1]["Paper B"], "right")


if __name__ == "__main__":
    unittest.main()
