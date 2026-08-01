"""Stage 4 - Indexing: captioned chunks -> Chroma vector store, tracked by
SQLRecordManager.

This is where images stop being images and become retrievable: what gets
embedded is chunk.text, which by now has each figure's caption folded in.
The embedding model never sees pixels - retrieval is pure text similarity -
and image_paths rides along as metadata for the answer step to render.

Change tracking (SQLRecordManager + LangChain's index() API):
  The record manager keeps a timestamped, hash-based record of every chunk
  already in the store. On re-index it compares incoming chunks against that
  record and reports num_added / num_updated / num_skipped / num_deleted.
  With cleanup="incremental" it also removes stale versions of a chunk whose
  content changed, keyed by source_id. Net effect: editing one chapter's PDF
  and re-running re-embeds only what actually changed, instead of rebuilding
  the whole book - and never leaves duplicate/orphaned vectors behind.

Output: vectorstore/ (Chroma), record_manager.sqlite (change-tracking DB)
"""

from __future__ import annotations

import json
import os

from .config import BookRef
from . import paths
from . import llm


def load_documents(book: BookRef) -> list:
    """Read captioned_chunks/{chapter}.json -> list[Document].

    Each Document's metadata carries a `source` key (chapter id) that the
    indexing API uses to group chunks for incremental cleanup, plus the
    figure paths (JSON-encoded, since Chroma metadata must be scalar).
    """
    from langchain_core.documents import Document

    chunks_dir = paths.captioned_chunks_dir(book)
    documents = []

    for fname in sorted(os.listdir(chunks_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(chunks_dir, fname)) as f:
            chunks = json.load(f)

        for i, c in enumerate(chunks):
            documents.append(Document(
                page_content=c["text"],
                metadata={
                    "source": c["chapter"],          # <- incremental-cleanup key
                    "chapter": c["chapter"],
                    "section": c["section"],
                    "chunk_index": i,
                    "pages": json.dumps(c["pages"]),
                    "image_paths": json.dumps(c["image_paths"]),
                },
            ))

    return documents


def get_vectorstore(book: BookRef, embeddings=None):
    """Open (or create) the Chroma collection for a book without indexing."""
    from langchain_chroma import Chroma

    embeddings = embeddings or llm.get_embeddings()
    return Chroma(
        persist_directory=paths.vectorstore_dir(book),
        embedding_function=embeddings,
        collection_name=paths.collection_name(book),
    )


def get_record_manager(book: BookRef):
    """SQLRecordManager backing this book's index, schema ensured."""
    # SQLRecordManager currently lives in langchain-community. Import
    # defensively so a future move (community is being sunset) is a one-line
    # change here rather than a crash.
    try:
        from langchain_community.indexes import SQLRecordManager
    except ImportError:  # pragma: no cover - fallback for internal path
        from langchain_community.indexes._sql_record_manager import SQLRecordManager

    db_path = paths.record_manager_db(book)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    rm = SQLRecordManager(
        namespace=paths.collection_name(book),
        db_url=f"sqlite:///{db_path}",
    )
    rm.create_schema()
    return rm


def build_index(
    book: BookRef,
    embeddings=None,
    cleanup: str = "incremental",
    progress=None,
) -> dict:
    """Index a book's captioned chunks into Chroma with change tracking.

    cleanup:
      "incremental" (default) - de-dupes and removes stale versions
        continuously; safe for repeated re-indexing after edits.
      "full" - also deletes anything in the store not present in this run
        (use when chapters may have been removed).
      None - insert only, no cleanup.

    Returns LangChain's index() report: {num_added, num_updated,
    num_skipped, num_deleted}.
    """
    from langchain_core.indexing import index

    if progress:
        progress(0.1, "Loading captioned chunks")
    documents = load_documents(book)

    if progress:
        progress(0.3, "Opening vector store and record manager")
    vectorstore = get_vectorstore(book, embeddings=embeddings)
    record_manager = get_record_manager(book)

    if progress:
        progress(0.5, f"Indexing {len(documents)} chunks (cleanup={cleanup})")
    result = index(
        documents,
        record_manager,
        vectorstore,
        cleanup=cleanup,
        source_id_key="source",
        key_encoder="sha256",  # collision-resistant; SHA-1 default is discouraged
    )

    if progress:
        progress(1.0, "Indexing complete")
    return result
