"""Filesystem path conventions for a book's source and generated artefacts.

Every stage of the pipeline reads the previous stage's output and writes its
own, and they agree on *where* those live purely through this module. Nothing
else in the codebase constructs a path by string concatenation, so adding a
new standard/subject never requires touching path logic anywhere.
"""

from __future__ import annotations

import os

from .config import DATA_ROOT, BookRef


def book_dir(book: BookRef) -> str:
    """Root folder for a book: {DATA_ROOT}/{standard}/{subject}/"""
    return os.path.join(DATA_ROOT, book.standard, book.subject)


def source_pdf_dir(book: BookRef) -> str:
    """Where the raw chapter PDFs live (same as book_dir)."""
    return book_dir(book)


def extracted_text_dir(book: BookRef) -> str:
    return os.path.join(book_dir(book), "extracted_text")


def extracted_images_dir(book: BookRef) -> str:
    return os.path.join(book_dir(book), "extracted_images")


def chunks_dir(book: BookRef) -> str:
    return os.path.join(book_dir(book), "chunks")


def captions_dir(book: BookRef) -> str:
    return os.path.join(book_dir(book), "captions")


def captioned_chunks_dir(book: BookRef) -> str:
    return os.path.join(book_dir(book), "captioned_chunks")


def vectorstore_dir(book: BookRef) -> str:
    return os.path.join(book_dir(book), "vectorstore")


def record_manager_db(book: BookRef) -> str:
    """SQLite file backing the SQLRecordManager for this book's index."""
    return os.path.join(book_dir(book), "record_manager.sqlite")


def collection_name(book: BookRef) -> str:
    """Chroma collection name for this book."""
    return f"ncert_{book.slug}"


def ensure_dirs(book: BookRef) -> None:
    """Create every output directory for a book if missing. Safe to call
    repeatedly."""
    for d in (
        extracted_text_dir(book),
        extracted_images_dir(book),
        chunks_dir(book),
        captions_dir(book),
        captioned_chunks_dir(book),
        vectorstore_dir(book),
    ):
        os.makedirs(d, exist_ok=True)


def discover_books() -> list[BookRef]:
    """Scan DATA_ROOT and return every {standard}/{subject} that contains at
    least one PDF. Powers the Streamlit UI's book picker."""
    books: list[BookRef] = []
    if not os.path.isdir(DATA_ROOT):
        return books
    for standard in sorted(os.listdir(DATA_ROOT)):
        std_path = os.path.join(DATA_ROOT, standard)
        if not os.path.isdir(std_path):
            continue
        for subject in sorted(os.listdir(std_path)):
            subj_path = os.path.join(std_path, subject)
            if not os.path.isdir(subj_path):
                continue
            if any(f.endswith(".pdf") for f in os.listdir(subj_path)):
                books.append(BookRef(standard=standard, subject=subject))
    return books
