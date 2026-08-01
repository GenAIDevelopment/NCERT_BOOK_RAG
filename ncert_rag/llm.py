"""Factory functions for the Gemini models the pipeline uses.

Centralising model construction here means:
  * the rest of the code asks for "a caption model" / "a chat model" /
    "an embeddings model" without knowing the provider or auth details;
  * switching the Vertex AI vs. Gemini-Developer-API backend is one config
    flag, handled in exactly one place;
  * tests can inject stubs by passing model objects directly to the
    functions that use them, bypassing these factories entirely.

Backend selection (langchain-google-genai, consolidated google-genai SDK):
  * Developer API (default): needs GOOGLE_API_KEY (or GEMINI_API_KEY).
  * Vertex AI: needs vertexai=True AND a GCP project. Crucially, vertexai=True
    ALONE is not enough -- without a project the client falls back to the
    Developer API path and then errors demanding an API key (the exact error
    this indirection is designed to prevent). So on the Vertex path we always
    pass project (+ location) and fail fast with a clear message if the project
    is missing, rather than surfacing a confusing "API key required" error.
"""

from __future__ import annotations

from . import config


def _vertex_kwargs() -> dict:
    """Extra kwargs required to actually activate the Vertex AI backend.

    Returns {} for the Developer-API path. For the Vertex path, returns
    project + location, raising a clear error if the project is unset."""
    if not config.USE_VERTEXAI:
        return {}
    if not config.GCP_PROJECT:
        raise RuntimeError(
            "Vertex AI backend selected (NCERT_USE_VERTEXAI=true) but no GCP "
            "project found. Set GOOGLE_CLOUD_PROJECT (or NCERT_GCP_PROJECT), "
            "and authenticate with `gcloud auth application-default login`. "
            "Alternatively set NCERT_USE_VERTEXAI=false and provide GOOGLE_API_KEY."
        )
    return {
        "vertexai": True,
        "project": config.GCP_PROJECT,
        "location": config.GCP_LOCATION,
    }


def get_caption_model():
    """Vision-capable Gemini model for image captioning."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.CAPTION_MODEL,
        temperature=0,
        **_vertex_kwargs(),
    )


def get_chat_model():
    """Gemini model for answering student questions in the RAG chain."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=config.CHAT_TEMPERATURE,
        **_vertex_kwargs(),
    )


def get_embeddings():
    """Gemini text-embedding model for indexing and query embedding.

    GoogleGenerativeAIEmbeddings takes the same backend kwargs as the chat
    model (project/location for Vertex)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        **_vertex_kwargs(),
    )