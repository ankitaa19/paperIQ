"""Prompt templates for summarization, RAG, comparison, and experiments."""

BASIC_QA = """Answer the question.
Question: {question}
"""

ROLE_QA = """You are a careful research assistant who explains machine-learning papers
in plain language for a graduate student.
Question: {question}
"""

STRUCTURED_QA = """Answer the question using this structure:
1. Direct answer
2. Supporting detail
3. What is uncertain
Question: {question}
"""

CONTEXT_QA = """Use only the context below. If the context does not contain the answer,
say: "Sufficient supporting evidence was not found in the uploaded research papers."
Do not add facts that are not in the context.

Context:
{context}

Question: {question}
"""

RAG_QA = """You are PaperIQ, a research assistant. Use only the retrieved excerpts.
A citation is not enough: every factual claim must be stated in the excerpt you cite.
If the excerpts name a paper but do not state the requested finding, do not fill the gap
from prior knowledge.
If the excerpts support only part of the question, answer that part and then write:
"The remaining part was not clearly specified in the retrieved excerpts."
If nothing in the excerpts answers the question, reply exactly:
"Sufficient supporting evidence was not found in the uploaded research papers."
Cite the file name and page for each claim, for example (lost_in_the_middle.pdf, p. 1).

Retrieved excerpts:
{context}

Question: {question}
"""

SUMMARY_PROMPT = """You are summarizing one research paper from the excerpts below.
Use only information present in the excerpts. Do not invent authors, numbers, or results.
Put statements that the excerpts explicitly support under each heading.
If you add a short interpretation, start that sentence with "Interpretation:".
If a section is not supported, write "Not clearly specified in the paper."

Return Markdown with these headings only:
## Research Objective
## Problem Statement
## Methodology
## Key Findings
## Limitations
## Conclusion

Paper file: {file_name}
Title hint: {title}

Excerpts:
{context}
"""

COMPARE_PROMPT = """Compare two research papers using only the excerpts below.
Paper A is {file_a}. Paper B is {file_b}. Do not swap them.
Do not invent experiments, numbers, or conclusions.
If a dimension is not supported by that paper's excerpts, set that paper's text to exactly:
"Not clearly specified in the available excerpts."
Include a page reference only when that page's excerpt supports the sentence.

Return one JSON object and no other text. Use these exact keys:
Research Objective, Problem Statement, Methodology, Models / Techniques, Experimental Results, Key Findings, Limitations, Conclusion.
Each value must be an object with keys paper_a and paper_b.

Paper A excerpts:
{context_a}

Paper B excerpts:
{context_b}
"""
