import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from src.pdf_processor import (
    EXPECTED_PAPERS,
    add_uploaded_pdf,
    list_papers,
    remove_uploaded_paper,
)
from src.summarizer import summarize_paper


class SummaryAndUploadTests(unittest.TestCase):
    def test_invalid_and_empty_pdfs_are_rejected(self):
        before = {p.name for p in list_papers()}
        with self.assertRaises(ValueError):
            add_uploaded_pdf("not-a-paper.pdf", b"this is not a pdf")
        empty = pymupdf.open()
        empty.new_page()
        raw = empty.tobytes()
        empty.close()
        with self.assertRaises(ValueError):
            add_uploaded_pdf("empty_upload.pdf", raw)
        self.assertEqual(before, {p.name for p in list_papers()})
        self.assertFalse((Path("data/papers") / "not-a-paper.pdf").exists())
        self.assertFalse((Path("data/papers") / "empty_upload.pdf").exists())

    def test_failed_generation_does_not_save(self):
        target = Path("data/processed/summaries/attention_is_all_you_need.json")
        before = target.read_bytes() if target.exists() else None
        with patch("src.summarizer.complete", side_effect=RuntimeError("groq down")):
            with self.assertRaises(RuntimeError):
                summarize_paper("attention_is_all_you_need.pdf")
        after = target.read_bytes() if target.exists() else None
        self.assertEqual(before, after)

    def test_sample_papers_cannot_be_removed(self):
        with self.assertRaises(ValueError):
            remove_uploaded_paper(EXPECTED_PAPERS[0])
        self.assertTrue((Path("data/papers") / EXPECTED_PAPERS[0]).exists())


if __name__ == "__main__":
    unittest.main()
