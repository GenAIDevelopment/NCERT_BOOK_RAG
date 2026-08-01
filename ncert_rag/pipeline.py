"""Ingestion orchestration: run all four stages for a book, in order.

    extract -> chunk -> caption -> index

Stages communicate through the filesystem (see paths.py), so each is
independently re-runnable: change the chunking config, re-run from chunk;
add a chapter PDF, re-run from extract. The index stage's SQLRecordManager
ensures a re-run only re-embeds what actually changed.
"""

from __future__ import annotations

from .config import BookRef
from . import paths
from . import extract as extract_stage
from . import chunk as chunk_stage
from . import caption as caption_stage
from . import index as index_stage


def ingest_book(
    book: BookRef,
    skip_files: list[str] | None = None,
    caption_limit: int | None = None,
    caption_model=None,
    embeddings=None,
    enable_ocr: bool = True,
    cleanup: str = "incremental",
    progress=None,
) -> dict:
    """Run the full ingestion pipeline for a book.

    progress: optional callable(fraction, message) for UI feedback; each stage
    gets an equal quarter of the 0..1 range.
    """
    paths.ensure_dirs(book)

    def stage_progress(base):
        if progress is None:
            return None
        return lambda frac, msg: progress(base + frac * 0.25, msg)

    extract_summary = extract_stage.extract_book(
        book, skip_files=skip_files, enable_ocr=enable_ocr, progress=stage_progress(0.0)
    )
    chunk_summary = chunk_stage.chunk_book(book, progress=stage_progress(0.25))
    caption_summary = caption_stage.caption_book(
        book, model=caption_model, limit=caption_limit, progress=stage_progress(0.5)
    )
    index_result = index_stage.build_index(
        book, embeddings=embeddings, cleanup=cleanup, progress=stage_progress(0.75)
    )

    if progress:
        progress(1.0, "Pipeline complete")

    return {
        "extract": extract_summary,
        "chunk": chunk_summary,
        "caption": caption_summary,
        "index": index_result,
    }
