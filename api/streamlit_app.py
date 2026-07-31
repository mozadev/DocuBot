"""
DocuBot AI - Streamlit interface.

Presentation only. Every question goes through the same ChatService the REST API
uses, so the UI and the API cannot drift apart in behaviour, guardrails or
tracing.

Streamlit was chosen over a React SPA for a reason worth stating: the interesting
part of this project is the retrieval pipeline, and a separate frontend would
have cost a day of build tooling to show the same three screens. The trade-off is
real and is discussed in the README.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from api.factory import build_container
from config.settings import settings

st.set_page_config(
    page_title="DocuBot AI",
    page_icon="/",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLES = """
<style>
    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    .db-title {
        font-size: 2rem; font-weight: 650; letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .db-subtitle {
        color: #6b7280; font-size: 0.95rem; margin-bottom: 1.75rem;
    }

    .db-source {
        border: 1px solid rgba(128,128,128,0.22);
        border-left: 3px solid #4f6bed;
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.86rem;
        line-height: 1.5;
    }
    .db-source-figure { border-left-color: #b45cc4; }
    .db-source-head {
        display: flex; justify-content: space-between;
        font-weight: 600; margin-bottom: 0.35rem;
    }
    .db-source-meta { color: #6b7280; font-weight: 400; font-size: 0.8rem; }
    .db-source-body { color: #4b5563; }

    .db-pill {
        display: inline-block; padding: 0.12rem 0.55rem;
        border-radius: 999px; font-size: 0.74rem; font-weight: 600;
        margin-right: 0.35rem;
    }
    .db-pill-high { background: rgba(22,163,74,0.14); color: #16a34a; }
    .db-pill-mid  { background: rgba(217,119,6,0.14); color: #d97706; }
    .db-pill-low  { background: rgba(220,38,38,0.14); color: #dc2626; }
    .db-pill-grey { background: rgba(107,114,128,0.16); color: #6b7280; }

    .db-empty {
        border: 1px dashed rgba(128,128,128,0.35);
        border-radius: 8px; padding: 2.5rem 1rem;
        text-align: center; color: #6b7280;
    }
</style>
"""


@st.cache_resource(show_spinner="Starting DocuBot...")
def get_container():
    return build_container()


def session_id() -> str:
    """A stable per-browser-session id, so histories never mix between users."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:12]}"
    return st.session_state.session_id


def confidence_pill(score: float) -> str:
    if score >= 0.45:
        cls, label = "db-pill-high", "high confidence"
    elif score >= 0.25:
        cls, label = "db-pill-mid", "medium confidence"
    else:
        cls, label = "db-pill-low", "low confidence"
    return f'<span class="db-pill {cls}">{label} · {score:.2f}</span>'


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            is_figure = src.get("content_type") == "image"
            page = src.get("page_number", 0)
            location = f"page {page}" if page else "—"
            kind = "figure" if is_figure else "text"

            st.markdown(
                f"""<div class="db-source {'db-source-figure' if is_figure else ''}">
                  <div class="db-source-head">
                    <span>{i}. {src.get('filename', 'unknown')}</span>
                    <span class="db-source-meta">{kind} · {location} · score {src.get('score', 0):.3f}</span>
                  </div>
                  <div class="db-source-body">{src.get('content', '')}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            img_path = src.get("image_path", "")
            if is_figure and img_path and os.path.exists(img_path):
                st.image(img_path, width=380)


def render_answer(entry: dict[str, Any]) -> None:
    with st.chat_message("assistant"):
        st.markdown(entry.get("answer", ""))

        badges = [confidence_pill(entry.get("confidence", 0.0))]
        if entry.get("cached"):
            badges.append('<span class="db-pill db-pill-grey">cached</span>')

        warnings = (entry.get("guardrail") or {}).get("warnings") or []
        if warnings:
            badges.append(
                f'<span class="db-pill db-pill-mid">{len(warnings)} guardrail warning(s)</span>'
            )

        st.markdown(" ".join(badges), unsafe_allow_html=True)

        if warnings:
            for w in warnings:
                st.caption(w)

        render_sources(entry.get("sources", []))


def documents_tab(container) -> None:
    doc_svc = container.documents
    extensions = [e.lstrip(".") for e in doc_svc.supported_extensions]

    uploaded = st.file_uploader(
        "Drop files here",
        type=extensions,
        accept_multiple_files=True,
        help=f"Supported: {', '.join(doc_svc.supported_extensions)}. "
             f"Max {settings.max_upload_mb} MB per file.",
    )

    if uploaded and st.button("Index documents", type="primary"):
        progress = st.progress(0.0)
        indexed = 0

        for i, file in enumerate(uploaded, 1):
            progress.progress(i / len(uploaded), text=f"Processing {file.name}...")
            ext = Path(file.name).suffix.lower()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                stats = doc_svc.process_and_index(tmp_path, original_filename=file.name)
                indexed += stats["total"]

                detail = f"{stats['text_chunks']} text chunks"
                if stats["image_chunks"]:
                    detail += f", {stats['image_chunks']} figures described"
                st.success(f"{file.name} — {detail}")
            except Exception as e:  # noqa: BLE001 - shown to the user per file
                st.error(f"{file.name} — {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        progress.empty()
        if indexed:
            st.info(f"{indexed} chunks indexed. Ask a question in the Chat tab.")

    st.divider()

    docs = doc_svc.list_documents()
    if docs:
        st.caption("Indexed documents")
        st.dataframe(
            [
                {
                    "File": d["filename"],
                    "Type": d.get("file_type", ""),
                    "Chunks": d["chunks"],
                    "Figures": d["images"],
                }
                for d in docs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown(
            '<div class="db-empty">No documents indexed yet.<br>'
            "Upload a PDF, DOCX, TXT or Markdown file to get started.</div>",
            unsafe_allow_html=True,
        )


def chat_tab(container) -> None:
    doc_svc, chat_svc = container.documents, container.chat

    if doc_svc.get_document_count() == 0:
        st.markdown(
            '<div class="db-empty">Nothing to search yet.<br>'
            "Index a document first, then come back and ask about it.</div>",
            unsafe_allow_html=True,
        )
        return

    st.session_state.setdefault("turns", [])

    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        render_answer(turn)

    question = st.chat_input("Ask something about your documents...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching your documents..."):
        response = chat_svc.ask_question(question, session_id=session_id())

    entry = response.to_dict()
    st.session_state.turns.append(entry)
    render_answer(entry)


def traces_tab(container) -> None:
    st.caption(
        "Every answer is traced. This is the same data the "
        "/api/v1/observability endpoints return."
    )

    analytics = container.tracer.get_analytics(session_id())
    if "message" in analytics:
        st.markdown(
            '<div class="db-empty">No traces yet.<br>'
            "Ask a question and its full execution trace appears here.</div>",
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", analytics["total_traces"])
    c2.metric("Avg latency", f"{analytics['avg_duration_ms']:.0f} ms")
    c3.metric("Errors", analytics["error_count"])
    c4.metric("Cache hit rate", f"{container.cache.get_stats()['hit_rate_pct']}%")

    st.divider()
    st.caption("Recent requests")

    for trace in container.tracer.get_tenant_traces(session_id(), limit=10):
        question = trace.get("metadata", {}).get("question", "(no question recorded)")
        with st.expander(f"{question[:80]} — {trace['total_duration_ms']:.0f} ms"):
            detail = container.tracer.get_trace(trace["id"])
            st.dataframe(
                [
                    {
                        "Step": s["name"],
                        "Type": s["span_type"],
                        "Duration (ms)": round(s["duration_ms"], 1),
                        "Output": str(s["output_data"])[:110],
                    }
                    for s in detail["spans"]
                ],
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)

    try:
        container = get_container()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not start DocuBot: {e}")
        st.caption("Check that OPENAI_API_KEY is set in your .env file.")
        return

    st.markdown('<div class="db-title">DocuBot AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="db-subtitle">Ask questions about your documents. '
        "Every answer is grounded in what you uploaded, and cites where it came from."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Knowledge base")
        st.metric("Indexed chunks", container.documents.get_document_count())

        st.subheader("Configuration")
        for label, value in {
            "Answering": settings.openai_model,
            "Embeddings": settings.embedding_model,
            "Figures": settings.vision_model if settings.enable_multimodal else "disabled",
            "Chunk size": f"{settings.chunk_size} / {settings.chunk_overlap} overlap",
        }.items():
            st.caption(f"**{label}** · {value}")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            container.chat.clear_memory(session_id())
            st.session_state.turns = []
            st.rerun()

        if st.button("Delete all documents", use_container_width=True):
            container.documents.clear_database()
            st.session_state.turns = []
            st.rerun()

        st.divider()
        st.caption(f"Session `{session_id()}`")

    chat, documents, traces = st.tabs(["Chat", "Documents", "Traces"])
    with chat:
        chat_tab(container)
    with documents:
        documents_tab(container)
    with traces:
        traces_tab(container)


if __name__ == "__main__":
    main()
