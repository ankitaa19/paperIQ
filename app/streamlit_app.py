"""PaperIQ research dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import groq_model
from src.evaluation import run_evaluation
from src.llm import resolve_model
from src.pdf_processor import (
    add_uploaded_pdf,
    is_sample,
    list_papers,
    load_processed,
    process_all,
    remove_uploaded_paper,
)
from src.paper_comparator import compare_papers
from src.prompts import BASIC_QA, CONTEXT_QA, RAG_QA, ROLE_QA, STRUCTURED_QA
from src.rag_pipeline import ask, compare_prompting_and_rag, run_prompt_modes
from src.retriever import search
from src.summarizer import summarize_paper
from src.vector_store import build_index, index_is_current

st.set_page_config(page_title="PaperIQ", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0B1020; color: #F8FAFC; }
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1180px; }
    section[data-testid="stSidebar"] {
        background: #111827; border-right: 1px solid #29364A;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stMetricValue"],
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #F8FAFC !important; }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: transparent; border-radius: 10px; padding: 0.28rem 0.45rem; margin-bottom: 0.12rem;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #1B2940; }
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #1B2940; box-shadow: inset 3px 0 0 #0098FF;
    }
    .pi-header h1 { font-size: 1.7rem; line-height: 1.2; margin: 0; letter-spacing: -0.03em; color: #F8FAFC; }
    .pi-sub { margin: 0.28rem 0 0; color: #F8FAFC; font-size: 1rem; }
    .pi-desc { margin: 0.25rem 0 0.85rem; color: #94A3B8; font-size: 0.92rem; max-width: 46rem; }
    .pi-section { margin: 1rem 0 0.45rem; color: #F8FAFC; font-size: 0.95rem; font-weight: 650; }
    .pi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.7rem; }
    .pi-card, .pi-paper, .pi-hit, .pi-note {
        background: #151F32; border: 1px solid #29364A; border-radius: 12px; padding: 0.85rem 0.95rem; min-width: 0;
    }
    .pi-card .k { color: #94A3B8; font-size: 0.75rem; margin: 0; }
    .pi-card .v { color: #F8FAFC; font-size: 1.3rem; font-weight: 650; margin: 0.22rem 0 0; }
    .pi-card .s, .pi-hit .meta, .pi-paper .file, .pi-paper .meta { color: #94A3B8; font-size: 0.82rem; margin: 0.15rem 0 0; }
    .pi-ok { color: #22C55E !important; }
    .pi-bad { color: #EF4444 !important; }
    .pi-library, .pi-results { display: flex; flex-direction: column; gap: 0.5rem; }
    .pi-paper { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr) 5.5rem 7.5rem; gap: 0.75rem; align-items: center; }
    .pi-paper .title, .pi-hit .title { margin: 0; color: #F8FAFC; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pi-paper .file { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .pi-badge { justify-self: start; border: 1px solid #1d4e89; color: #7dd3fc; background: rgba(0,152,255,0.12); border-radius: 999px; padding: 0.12rem 0.55rem; font-size: 0.75rem; }
    .pi-hit { margin-bottom: 0.55rem; }
    div[data-testid="stButton"] button, div[class*="st-key-dash_"] button {
        border-radius: 10px; border: 1px solid #29364A; background: #1B2940; color: #F8FAFC;
    }
    div[data-testid="stButton"] button:hover, div[class*="st-key-dash_"] button:hover { border-color: #38BDF8; color: #F8FAFC; }
    button[kind="primary"] {
        background: #0098FF !important; border-color: #0098FF !important; color: #041018 !important; font-weight: 650;
    }
    button[kind="primary"]:hover { background: #38BDF8 !important; border-color: #38BDF8 !important; }
    [data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #38BDF8 !important; border-radius: 16px !important; background: #151F32 !important;
        min-height: 220px; padding: 1.5rem 1rem !important;
        display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.55rem;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: #0098FF !important; background: #1B2940 !important; }
    [data-testid="stFileUploaderDropzone"]::before {
        content: ""; width: 42px; height: 42px;
        background: center / contain no-repeat url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2338BDF8' stroke-width='1.6'%3E%3Cpath d='M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 2v6h6'/%3E%3Cpath d='M12 18v-6'/%3E%3Cpath d='M9.5 14.5 12 12l2.5 2.5'/%3E%3C/svg%3E");
    }
    [data-testid="stFileUploaderDropzone"] > span { order: 3; }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        order: 2; display: flex !important; flex-direction: column !important; align-items: center !important; text-align: center;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "Drag and drop your research papers here"; color: #F8FAFC; font-size: 1.05rem; font-weight: 650;
    }
    [data-testid="stFileUploaderDropzoneInstructions"]::after { content: "PDF only · multiple files"; color: #94A3B8; font-size: 0.85rem; }
    [data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0 !important; line-height: 0 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] span::after {
        content: "or click Browse Files"; font-size: 0.92rem !important; line-height: 1.4 !important; color: #94A3B8;
    }
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] p { font-size: 0 !important; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] p::after {
        content: "Browse Files"; font-size: 0.95rem; color: #F8FAFC;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: #0098FF !important; color: #041018 !important; border: none !important; border-radius: 8px !important;
    }
    [data-testid="stDataFrame"] { border: 1px solid #29364A; border-radius: 12px; overflow: hidden; }
    @media (max-width: 900px) {
        .pi-grid { grid-template-columns: 1fr 1fr; }
        .pi-paper { grid-template-columns: minmax(0, 1fr); }
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    }
    @media (max-width: 520px) { .pi-grid { grid-template-columns: 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Dashboard",
    "Upload Research Papers",
    "Paper Summarization",
    "Semantic Search",
    "Ask Research Assistant",
    "Compare Research Papers",
    "Evaluation and Experiments",
]


def paper_names() -> list[str]:
    return [p.name for p in list_papers()]


_NAV_ICON = {
    "Dashboard": "▣",
    "Upload Research Papers": "↑",
    "Paper Summarization": "≡",
    "Semantic Search": "⌕",
    "Ask Research Assistant": "✎",
    "Compare Research Papers": "⇄",
    "Evaluation and Experiments": "▤",
}
_SUMMARY_SECTIONS = [
    "Research Objective",
    "Problem Statement",
    "Methodology",
    "Key Findings",
    "Limitations",
    "Conclusion",
]


def _esc(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def page_header(title: str, description: str) -> None:
    st.markdown(
        f'<header class="pi-header"><h1>{_esc(title)}</h1><p class="pi-desc">{_esc(description)}</p></header>',
        unsafe_allow_html=True,
    )


def render_summary_sections(text: str) -> None:
    chunks = {}
    for part in ("\n" + text).split("\n## "):
        body = part.strip()
        if not body:
            continue
        title, _, rest = body.partition("\n")
        title = title.replace("#", "").strip()
        if title in _SUMMARY_SECTIONS:
            chunks[title] = rest.strip()
    if len(chunks) < 4:
        st.markdown(text)
        return
    for title in _SUMMARY_SECTIONS:
        section = chunks.get(title)
        if not section:
            continue
        st.markdown(f'<p class="pi-section">{title}</p>', unsafe_allow_html=True)
        st.markdown(section)


def sidebar() -> str:
    st.sidebar.markdown("## PaperIQ")
    st.sidebar.caption("AI-powered research paper assistant")
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = PAGES[0]
    page = st.sidebar.radio(
        "Navigate",
        PAGES,
        key="nav_page",
        label_visibility="collapsed",
        format_func=lambda name: f"{_NAV_ICON[name]}  {name}",
    )
    st.sidebar.metric("Papers in library", len(paper_names()))
    return page


def _open_page(page: str) -> None:
    st.session_state["nav_page"] = page


def _groq_status() -> tuple[str, str]:
    """Return (status, model). The live check runs once per browser session."""
    cached = st.session_state.get("groq_status")
    if cached:
        return cached
    configured = groq_model()
    try:
        model = resolve_model()
        cached = ("Connected", model or configured)
    except Exception as exc:  # noqa: BLE001
        cached = (f"Unavailable: {exc}", configured)
    st.session_state["groq_status"] = cached
    return cached


def _library_rows() -> list[dict]:
    rows = []
    for name in paper_names():
        record = load_processed(name) or {}
        pages = record.get("extracted_pages")
        rows.append(
            {
                "title": record.get("title") or "Title not extracted",
                "file": name,
                "pages": pages if pages is not None else "—",
                "status": "Extracted" if record.get("pages") else "Not extracted",
            }
        )
    return rows


def page_dashboard() -> None:
    rows = _library_rows()
    page_total = sum(row["pages"] for row in rows if isinstance(row["pages"], int))
    try:
        index_ok = index_is_current()
        index_label = "Available" if index_ok else "Out of date"
    except Exception as exc:  # noqa: BLE001
        index_ok = False
        index_label = f"Unavailable: {exc}"
    model_status, model_name = _groq_status()
    model_ok = model_status == "Connected"
    st.markdown(
        """
        <header class="pi-header">
          <h1>PaperIQ</h1>
          <p class="pi-sub">Your AI-powered research paper assistant</p>
          <p class="pi-desc">Upload, summarize, search, and compare research papers using AI-powered insights.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    left.button(
        "Upload Research Papers",
        key="dash_primary_upload",
        on_click=_open_page,
        args=("Upload Research Papers",),
        use_container_width=True,
    )
    right.button(
        "Ask Research Assistant",
        key="dash_primary_ask",
        on_click=_open_page,
        args=("Ask Research Assistant",),
        use_container_width=True,
    )
    st.markdown('<p class="pi-section">Research overview</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="pi-grid">
          <article class="pi-card"><p class="k">Total Research Papers</p><p class="v">{len(rows)}</p></article>
          <article class="pi-card"><p class="k">Total Pages Processed</p><p class="v">{page_total}</p></article>
          <article class="pi-card"><p class="k">Search Index Status</p><p class="v {'pi-ok' if index_ok else 'pi-bad'}">{index_label}</p></article>
          <article class="pi-card"><p class="k">AI Model Status</p><p class="v {'pi-ok' if model_ok else 'pi-bad'}">{model_status}</p><p class="s">{model_name}</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="pi-section">Research paper library</p>', unsafe_allow_html=True)
    if not rows:
        st.info("No PDFs found in data/papers.")
    else:
        cards = []
        for row in rows:
            title = str(row["title"]).replace("&", "&amp;").replace("<", "&lt;")
            file_name = str(row["file"]).replace("&", "&amp;").replace("<", "&lt;")
            cards.append(
                f'<article class="pi-paper" title="{title}">'
                f'<p class="title">{title}</p><p class="file">{file_name}</p>'
                f'<p class="meta">{row["pages"]} pages</p><span class="pi-badge">{row["status"]}</span>'
                f"</article>"
            )
        st.markdown(f'<div class="pi-library">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.markdown('<p class="pi-section">Quick actions</p>', unsafe_allow_html=True)
    actions = [
        ("Upload Papers", "Upload Research Papers"),
        ("Generate Summary", "Paper Summarization"),
        ("Semantic Search", "Semantic Search"),
        ("Ask Research Assistant", "Ask Research Assistant"),
        ("Compare Papers", "Compare Research Papers"),
        ("Evaluation & Experiments", "Evaluation and Experiments"),
    ]
    action_cols = st.columns(3)
    for index, (label, target) in enumerate(actions):
        action_cols[index % 3].button(
            label,
            key=f"dash_action_{index}",
            on_click=_open_page,
            args=(target,),
            use_container_width=True,
        )


def page_upload() -> None:
    page_header(
        "Upload research papers",
        "Add PDFs beside the sample papers. Extraction, duplicate checks, and indexing stay the same.",
    )
    uploads = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploads and st.button("Save uploads", type="primary"):
        added = []
        for upload in uploads:
            try:
                record = add_uploaded_pdf(upload.name, upload.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(f"{upload.name}: {exc}")
                continue
            added.append(record["file_name"])
            st.success(f"Saved and extracted {record['file_name']} ({record['extracted_pages']} pages).")
        if added:
            with st.spinner("Updating the search index..."):
                try:
                    info = build_index(force=False)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"The PDF was saved, but the search index was not updated: {exc}")
                else:
                    st.caption(f"Search index {info['status']}: {info['chunks']} chunks.")
    uploaded = [name for name in paper_names() if not is_sample(name)]
    if uploaded:
        st.subheader("Uploaded papers")
        remove_name = st.selectbox("Remove an uploaded paper", uploaded)
        if st.button("Remove uploaded paper"):
            try:
                remove_uploaded_paper(remove_name)
                info = build_index(force=True)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
            else:
                st.success(f"Removed {remove_name}. Search index rebuilt with {info['chunks']} chunks.")
                st.session_state.pop("shown_summary", None)
                st.session_state["summary_for"] = None
    st.subheader("Extract text")
    if st.button("Extract all PDFs"):
        with st.spinner("Reading PDFs..."):
            try:
                rows = process_all(force=False)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success("Extraction finished. Cached papers were not rewritten.")
    if st.button("Build search index"):
        with st.spinner("Embedding chunks. The first run downloads a small model."):
            try:
                info = build_index(force=False)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        st.success(f"Index {info['status']}: {info['chunks']} chunks.")


def page_summary() -> None:
    page_header(
        "Paper summarization",
        "Choose a paper and generate a six-section summary. Nothing is generated until you click the button.",
    )
    names = paper_names()
    if not names:
        st.info("Add a paper before requesting a summary.")
        return
    choice = st.selectbox("Paper", names)
    if st.button("Generate summary", type="primary"):
        with st.spinner("Writing a structured summary from the extracted text..."):
            try:
                payload = summarize_paper(choice)
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("shown_summary", None)
                st.session_state["summary_for"] = None
                st.error(str(exc))
                payload = None
        if payload:
            st.session_state["shown_summary"] = payload
            st.session_state["summary_for"] = choice
    payload = st.session_state.get("shown_summary")
    if payload and st.session_state.get("summary_for") == choice:
        st.caption(
            f"{payload.get('status', 'saved')} · {payload.get('model', '')} · {payload.get('seconds', '')}s"
        )
        render_summary_sections(payload["summary"])
    else:
        st.info("Select a paper and click Generate summary.")


def page_search() -> None:
    page_header(
        "Semantic search",
        "Search the indexed passages. Results keep the current ranking, page, and similarity score.",
    )
    query = st.text_input("Query", placeholder="scaled dot-product attention")
    k = st.slider("Results", 1, 8, 5)
    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            try:
                hits = search(query, k=k)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        if not hits:
            st.info("No matching passages.")
            return
        for hit in hits:
            record = load_processed(hit["file_name"]) or {}
            title = record.get("title") or hit["file_name"]
            st.markdown(
                f'<article class="pi-hit"><p class="title" title="{_esc(title)}">{_esc(title)}</p>'
                f'<p class="meta">{_esc(hit["file_name"])} · page {hit["page"]} · score {hit["score"]:.3f}</p></article>',
                unsafe_allow_html=True,
            )
            st.write(hit["text"])


def page_ask() -> None:
    page_header(
        "Ask the research assistant",
        "Ask one question. Basic, role, and structured prompts do not retrieve passages. Context and RAG do.",
    )
    question = st.text_area("Question", height=90, placeholder="What problem does Self-RAG address?")
    mode = st.selectbox(
        "Prompt style",
        ["rag", "context", "role", "structured", "basic"],
        format_func=lambda m: {
            "rag": "RAG (retrieved paper context)",
            "context": "Context-aware",
            "role": "Role prompt, no retrieval",
            "structured": "Structured prompt, no retrieval",
            "basic": "Basic prompt, no retrieval",
        }[m],
    )
    if st.button("Answer", type="primary") and question.strip():
        with st.spinner("Generating..."):
            try:
                result = ask(question, mode=mode)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        st.session_state["answer"] = result
    result = st.session_state.get("answer")
    if not result:
        st.info("Ask a question about the papers in the library.")
        return
    st.caption(f"{result['mode']} · {result['model']} · {result['seconds']}s")
    if result["unsupported"]:
        st.warning("The answer is marked as lacking sufficient support in the retrieved passages.")
    st.markdown(result["answer"])
    if result["sources"]:
        st.markdown('<p class="pi-section">Sources</p>', unsafe_allow_html=True)
        for source in result["sources"]:
            record = load_processed(source["file_name"]) or {}
            title = record.get("title") or source["file_name"]
            st.markdown(
                f'<article class="pi-hit"><p class="title">{_esc(title)}</p>'
                f'<p class="meta">{_esc(source["file_name"])} · page {source["page"]} · score {source["score"]}</p></article>',
                unsafe_allow_html=True,
            )
            st.caption(source["text"][:500])


def page_compare() -> None:
    page_header(
        "Compare research papers",
        "Compare two papers across the eight research dimensions. The table is built from the validated comparison.",
    )
    names = paper_names()
    if len(names) < 2:
        st.info("At least two extracted papers are required.")
        return
    left, right = st.columns(2)
    file_a = left.selectbox("Paper A", names, index=0)
    file_b = right.selectbox("Paper B", names, index=1)
    if st.button("Compare", type="primary"):
        with st.spinner("Comparing the two papers..."):
            try:
                result = compare_papers(file_a, file_b)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        st.session_state["comparison"] = result
    result = st.session_state.get("comparison")
    if result:
        a, b = st.columns(2)
        a.markdown(f"**{result['title_a']}**")
        a.caption(result["file_a"])
        b.markdown(f"**{result['title_b']}**")
        b.caption(result["file_b"])
        st.caption(f"{result['model']} · {result['seconds']}s")
        if result.get("rows"):
            st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True, hide_index=True)
        else:
            st.markdown(result.get("comparison", ""))


def page_eval() -> None:
    page_header(
        "Evaluation and experiments",
        "These checks run against the live index and Groq. Figures appear only after a run; they are not preset scores.",
    )
    with st.expander("Prompt templates"):
        st.markdown("**Basic**")
        st.code(BASIC_QA)
        st.markdown("**Role**")
        st.code(ROLE_QA)
        st.markdown("**Structured**")
        st.code(STRUCTURED_QA)
        st.markdown("**Context-aware**")
        st.code(CONTEXT_QA)
        st.markdown("**RAG**")
        st.code(RAG_QA)

    mode_question = st.text_input(
        "Run one question through all five prompt modes",
        value="What mechanism does the Transformer use instead of recurrence?",
    )
    if st.button("Run five prompt modes", type="primary"):
        with st.spinner("Running five prompts. Only context and RAG retrieve passages."):
            try:
                modes = run_prompt_modes(mode_question)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                modes = []
        for item in modes:
            st.markdown(f"**{item['mode']}** · {item['seconds']}s · sources: {len(item['sources'])}")
            st.write(item["answer"])

    question = st.text_input("Compare prompting-only and RAG on one question", value="What is a Transformer?")
    if st.button("Compare prompting-only and RAG", type="primary"):
        with st.spinner("Asking Groq twice..."):
            try:
                pair = compare_prompting_and_rag(question)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                pair = None
        if pair:
            left, right = st.columns(2)
            left.subheader("Prompting only")
            left.caption(f"{pair['prompting_only']['seconds']}s")
            left.write(pair["prompting_only"]["answer"])
            right.subheader("RAG")
            right.caption(f"{pair['rag']['seconds']}s")
            right.write(pair["rag"]["answer"])
            if pair["rag"]["sources"]:
                right.caption(
                    "; ".join(f"{s['file_name']} p.{s['page']}" for s in pair["rag"]["sources"][:4])
                )

    if st.button("Run evaluation cases", type="primary"):
        with st.spinner("Running retrieval and both answer modes. This makes several API calls."):
            try:
                report = run_evaluation()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        for case in report["cases"]:
            st.markdown(f"### {case['id']} ({case['kind']})")
            st.write(case["question"])
            st.caption(f"Reference: {case['reference']}")
            checks = case["checks"]
            st.write(
                f"Deterministic retrieval check: {checks['retrieval_pass']}. "
                f"Cited expected file: {checks['cited_expected_file']}. "
                f"Refusal check: {checks['refusal_pass']}. "
                f"Correctness: {checks['correctness']}"
            )
            retrieval = case["retrieval"]
            st.write(
                f"Retrieval {retrieval['seconds']}s · top files: {', '.join(retrieval['top_files']) or 'none'} · "
                f"expected file found: {retrieval['expected_file_retrieved']}"
            )
            left, right = st.columns(2)
            left.markdown("**Prompting only**")
            left.caption(f"{case['prompting_only']['seconds']}s")
            left.write(case["prompting_only"]["answer"])
            right.markdown("**RAG**")
            right.caption(
                f"{case['rag']['seconds']}s · insufficient evidence flag: {case['rag']['unsupported']}"
            )
            right.write(case["rag"]["answer"])
            if case["rag"]["sources"]:
                right.caption(
                    "; ".join(f"{s['file_name']} p.{s['page']}" for s in case["rag"]["sources"][:4])
                )


def main() -> None:
    page = sidebar()
    model_status, model_name = _groq_status()
    if model_status == "Connected":
        st.sidebar.caption(f"Groq model: {model_name}")
    else:
        st.sidebar.error(f"Groq is not ready: {model_status.removeprefix('Unavailable: ')}")
    {
        "Dashboard": page_dashboard,
        "Upload Research Papers": page_upload,
        "Paper Summarization": page_summary,
        "Semantic Search": page_search,
        "Ask Research Assistant": page_ask,
        "Compare Research Papers": page_compare,
        "Evaluation and Experiments": page_eval,
    }[page]()


if __name__ == "__main__":
    main()
