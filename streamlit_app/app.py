"""Streamlit UI for the NCERT Book RAG system.

Two pages:
  * Index a Book  - pick a standard/subject, run the ingestion pipeline with
    live progress, see the SQLRecordManager change report.
  * Ask a Teacher - chat interface over a built index; answers cite sections
    and render the relevant figures inline.

Run:  streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import os
import sys

# make the ncert_rag package importable when run via `streamlit run`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from ncert_rag import BookRef, discover_books, TeacherRAG
from ncert_rag import pipeline
from ncert_rag import paths


st.set_page_config(page_title="NCERT Book RAG", page_icon="📚", layout="wide")


def _book_label(b: BookRef) -> str:
    return f"Class {b.standard} · {b.subject}"


def page_index():
    st.header("📥 Index a Book")
    st.caption(
        "Run the extract → chunk → caption → index pipeline. Re-indexing only "
        "re-embeds what changed, thanks to SQLRecordManager change tracking."
    )

    books = discover_books()
    if not books:
        st.warning(
            f"No books found under `{paths.DATA_ROOT if hasattr(paths, 'DATA_ROOT') else 'data root'}`. "
            "Add PDFs under data/standard/{class}/{subject}/ and reload."
        )
        return

    book = st.selectbox("Book", books, format_func=_book_label)

    col1, col2 = st.columns(2)
    with col1:
        cleanup = st.selectbox(
            "Cleanup mode",
            ["incremental", "full", "none"],
            help="incremental: de-dupe & remove stale versions continuously. "
                 "full: also delete anything not in this run. none: insert only.",
        )
        enable_ocr = st.checkbox("OCR text baked into images", value=True)
    with col2:
        caption_limit = st.number_input(
            "Caption limit per chapter (0 = no limit)",
            min_value=0, value=0,
            help="Cap newly-captioned images per chapter for a cheap dry run.",
        )
        skip_answer_key = st.checkbox("Skip answer-key PDF (gegp1ps.pdf)", value=True)

    if st.button("▶️ Run indexing pipeline", type="primary"):
        bar = st.progress(0.0, text="Starting…")

        def on_progress(frac, msg):
            bar.progress(min(frac, 1.0), text=msg)

        with st.spinner("Running pipeline…"):
            result = pipeline.ingest_book(
                book,
                skip_files=["gegp1ps.pdf"] if skip_answer_key else [],
                caption_limit=(caption_limit or None),
                enable_ocr=enable_ocr,
                cleanup=(None if cleanup == "none" else cleanup),
                progress=on_progress,
            )

        st.success("Pipeline complete.")
        rep = result["index"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Added", rep["num_added"])
        c2.metric("Updated", rep["num_updated"])
        c3.metric("Skipped", rep["num_skipped"])
        c4.metric("Deleted", rep["num_deleted"])

        with st.expander("Per-chapter detail"):
            st.write("**Chunking**", result["chunk"])
            st.write("**Captioning**", result["caption"])


def page_chat():
    st.header("👩‍🏫 Ask a Teacher")
    st.caption("Ask a question about the book; answers come only from the textbook.")

    books = discover_books()
    if not books:
        st.warning("No indexed books available. Index one first.")
        return

    book = st.selectbox("Book", books, format_func=_book_label, key="chat_book")

    # (re)build the RAG object when the book changes
    if st.session_state.get("_rag_book") != book.slug:
        try:
            st.session_state._rag = TeacherRAG(book)
            st.session_state._rag_book = book.slug
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Could not open the index for this book: {e}")
            return

    for m in st.session_state.get("messages", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for img in m.get("images", []):
                if os.path.exists(img):
                    st.image(img, width=280)

    question = st.chat_input("e.g. Why is one lakh a large number?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state._rag.answer(question)
            st.markdown(result.answer)

            shown = [p for p in result.images if os.path.exists(p)]
            if shown:
                st.caption("Relevant figures from the book:")
                cols = st.columns(min(len(shown), 3))
                for i, img in enumerate(shown):
                    cols[i % len(cols)].image(img, use_container_width=True)

            if result.sources:
                srcs = ", ".join(
                    f"{s['chapter']} · {s['section']}" for s in result.sources
                )
                st.caption(f"Sources: {srcs}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "images": shown,
        })


def _backend_status():
    """Show which Gemini backend the app detected, so misconfiguration is
    visible immediately rather than surfacing as an opaque model error."""
    from ncert_rag import config

    if config.USE_VERTEXAI:
        if config.GCP_PROJECT:
            st.sidebar.success(f"Backend: Vertex AI\nProject: {config.GCP_PROJECT}")
        else:
            st.sidebar.error(
                "Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set. "
                "The app can't see your project — check that your env vars are "
                "in a .env file at the project root, or exported in the same "
                "shell you launched `streamlit run` from."
            )
    else:
        import os
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            st.sidebar.success("Backend: Gemini Developer API (API key found)")
        else:
            st.sidebar.error(
                "No backend configured. Either set GOOGLE_GENAI_USE_VERTEXAI=true "
                "+ GOOGLE_CLOUD_PROJECT (Vertex), or GOOGLE_API_KEY (Developer "
                "API). Put them in a .env file at the project root so the app "
                "picks them up regardless of how it was launched."
            )


def main():
    st.sidebar.title("📚 NCERT Book RAG")
    page = st.sidebar.radio("Page", ["Ask a Teacher", "Index a Book"])
    st.sidebar.divider()
    _backend_status()
    st.sidebar.divider()
    st.sidebar.caption(
        "Gemini-powered multimodal RAG over NCERT textbooks. "
        "Captions make figures retrievable; SQLRecordManager tracks changes."
    )

    if page == "Index a Book":
        page_index()
    else:
        page_chat()


if __name__ == "__main__":
    main()