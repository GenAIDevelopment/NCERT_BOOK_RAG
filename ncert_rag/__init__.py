"""NCERT Book RAG - a multimodal RAG pipeline over NCERT textbooks.

Public API:
    BookRef                  - identify a book by standard + subject
    ingest_book              - run the full extract->chunk->caption->index pipeline
    TeacherRAG               - query a built index, answering as a teacher
    discover_books           - list books available under the data root
"""

from .config import BookRef
from .pipeline import ingest_book
from .rag import TeacherRAG, RAGAnswer
from .paths import discover_books

__all__ = [
    "BookRef",
    "ingest_book",
    "TeacherRAG",
    "RAGAnswer",
    "discover_books",
]

__version__ = "1.0.0"
