"""The RAG chain: retrieve relevant chunks, answer as a patient teacher.

The system prompt casts the model as a teacher explaining to a student, and
constrains it to answer from retrieved context only (no outside facts, no
guessing) - important for a study aid, where a confident wrong answer is worse
than "that isn't covered in this chapter."

answer() returns both the prose answer and the figure paths from the retrieved
chunks, so the UI can render the relevant diagrams alongside the explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import BookRef, RETRIEVER_K
from . import index as index_module
from . import llm


TEACHER_SYSTEM_PROMPT = """You are a warm, patient teacher helping a school \
student understand their NCERT textbook. Explain concepts clearly, in simple \
language suited to the student's grade level, building intuition with concrete \
examples before formal statements.

Rules:
- Answer ONLY using the provided context from the textbook. Do not add facts \
from outside it.
- Do NOT volunteer extra rules, notation conventions, or details the context \
does not state (e.g. exactly where commas fall in a number). If you are \
tempted to explain a convention the context doesn't spell out, either quote \
what the context does say or leave it out -- a confident wrong detail is worse \
than a shorter correct answer.
- If the context does not contain the answer, say so plainly and suggest which \
section or chapter the student might look at instead. Never invent an answer.
- When the context includes a figure description, refer to it naturally \
("as the diagram shows...") so the student knows to look at it.
- Be encouraging. If a question reveals a misunderstanding, gently correct it.
- Keep explanations focused; don't pad. End with a short check-for-understanding \
question when it would help the student learn."""

TEACHER_USER_PROMPT = """Context from the textbook:
{context}

Student's question: {question}

Explain the answer as their teacher."""


@dataclass
class RAGAnswer:
    """Structured result of a RAG query."""

    answer: str
    images: list = field(default_factory=list)
    sources: list = field(default_factory=list)


def _format_context(docs) -> str:
    parts = []
    for d in docs:
        parts.append(f"[{d.metadata.get('chapter')} - {d.metadata.get('section')}]\n{d.page_content}")
    return "\n\n".join(parts)


def _collect_images(docs) -> list:
    """Decode the JSON-encoded image_paths metadata back into a deduped list."""
    seen = {}
    for d in docs:
        raw = d.metadata.get("image_paths", "[]")
        try:
            for p in json.loads(raw):
                seen.setdefault(p, None)
        except (json.JSONDecodeError, TypeError):
            continue
    return list(seen.keys())


class TeacherRAG:
    """A reusable RAG interface for one book. Construct once, query many times."""

    def __init__(self, book: BookRef, chat_model=None, embeddings=None, k: int = RETRIEVER_K):
        from .config import (
            RETRIEVER_SCORE_THRESHOLD,
            USE_RERANKER,
            RERANKER_MODEL,
            RERANK_FETCH_K,
        )

        self.book = book
        self.k = k
        self._chat = chat_model or llm.get_chat_model()
        self._vectorstore = index_module.get_vectorstore(book, embeddings=embeddings)

        if USE_RERANKER:
            # Two-stage: fetch a wide net via embeddings, then a cross-encoder
            # reranks (query, chunk) pairs and keeps the top k. This is what
            # most directly lifts Contextual Relevancy/Precision -- the chunks
            # that survive are the most on-topic, not just the nearest in
            # embedding space. Imports are local so the reranker deps are only
            # needed when the feature is turned on.
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            base = self._vectorstore.as_retriever(search_kwargs={"k": RERANK_FETCH_K})
            cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
            compressor = CrossEncoderReranker(model=cross_encoder, top_n=k)
            self._retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base
            )
        elif RETRIEVER_SCORE_THRESHOLD > 0:
            # Drop marginally-relevant chunks before they reach the LLM context.
            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": RETRIEVER_SCORE_THRESHOLD},
            )
        else:
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})

    def answer(self, question: str) -> RAGAnswer:
        from langchain_core.messages import SystemMessage, HumanMessage

        docs = self._retriever.invoke(question)
        context = _format_context(docs)

        messages = [
            SystemMessage(content=TEACHER_SYSTEM_PROMPT),
            HumanMessage(content=TEACHER_USER_PROMPT.format(context=context, question=question)),
        ]
        response = self._chat.invoke(messages).content.strip()

        return RAGAnswer(
            answer=response,
            images=_collect_images(docs),
            sources=[
                {"chapter": d.metadata.get("chapter"), "section": d.metadata.get("section")}
                for d in docs
            ],
        )