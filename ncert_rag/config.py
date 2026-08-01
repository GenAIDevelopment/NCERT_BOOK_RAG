"""Central configuration for the NCERT RAG pipeline.

Everything tunable lives here so the rest of the codebase never hardcodes a
model name, a path convention, or a threshold. Values can be overridden with
environment variables (handy for switching Gemini backends, or pointing at a
different data root in a container) without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# Load a .env file if present, BEFORE reading any environment variables below.
# This is what makes `streamlit run` / notebooks / subprocesses see the same
# GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT you set for a plain shell --
# a very common source of "works in my terminal but not in the app" errors.
# python-dotenv is optional; if it isn't installed we simply skip loading and
# rely on the real environment.
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:  # pragma: no cover - dotenv is optional
    pass


# --------------------------------------------------------------------------- #
# Filesystem conventions
# --------------------------------------------------------------------------- #
# Root under which every book lives, organised as:
#   {DATA_ROOT}/{standard}/{subject}/{chapter}.pdf
# with generated artefacts co-located in sibling folders (extracted_text/,
# extracted_images/, chunks/, captions/, captioned_chunks/, vectorstore/,
# record_manager.sqlite). Keeping generation output next to its source means a
# whole book's state is one directory you can inspect, back up, or delete.
DATA_ROOT = os.environ.get("NCERT_DATA_ROOT", "./data/standard")


# --------------------------------------------------------------------------- #
# Image extraction thresholds (see extract.py)
# --------------------------------------------------------------------------- #
DECORATIVE_MIN_DIM = int(os.environ.get("NCERT_DECORATIVE_MIN_DIM", 2000))
DECORATIVE_MAX_BYTES = int(os.environ.get("NCERT_DECORATIVE_MAX_BYTES", 12000))
MIN_CONTENT_BYTES = int(os.environ.get("NCERT_MIN_CONTENT_BYTES", 8000))

# OCR (text baked into image pixels)
OCR_MIN_CONFIDENCE = int(os.environ.get("NCERT_OCR_MIN_CONFIDENCE", 60))
OCR_MIN_CHARS = int(os.environ.get("NCERT_OCR_MIN_CHARS", 8))

# Vector-drawn figure detection (flowcharts / box diagrams)
VECTOR_FIG_MIN_AREA_FRAC = float(os.environ.get("NCERT_VECTOR_FIG_MIN_AREA_FRAC", 0.02))
VECTOR_FIG_MAX_AREA_FRAC = float(os.environ.get("NCERT_VECTOR_FIG_MAX_AREA_FRAC", 0.85))
VECTOR_FIG_DPI = int(os.environ.get("NCERT_VECTOR_FIG_DPI", 150))


# --------------------------------------------------------------------------- #
# Chunking (see chunk.py)
# --------------------------------------------------------------------------- #
MAX_CHUNK_CHARS = int(os.environ.get("NCERT_MAX_CHUNK_CHARS", 1500))
CHUNK_OVERLAP = int(os.environ.get("NCERT_CHUNK_OVERLAP", 150))


# --------------------------------------------------------------------------- #
# Google Gemini models
# --------------------------------------------------------------------------- #
# The three model roles the pipeline uses. All default to Gemini, per the
# initial build requirement. Swapping to another provider means changing the
# factory functions in llm.py, not touching business logic.
#
# Defaults are conservative, widely-available model names. Newer models
# (e.g. gemini-3.x-flash) may be available in your project/region -- override
# via the env vars below. The embedding model in particular must be enabled in
# your Vertex region, or indexing will fail with a model-availability error.
CAPTION_MODEL = os.environ.get("NCERT_CAPTION_MODEL", "gemini-2.5-flash")
CHAT_MODEL = os.environ.get("NCERT_CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.environ.get("NCERT_EMBEDDING_MODEL", "models/text-embedding-004")

# Whether to route through Vertex AI (enterprise/GCP billing) or the Gemini
# Developer API (simple API key). True -> Vertex AI.
#
# We honour BOTH our own NCERT_USE_VERTEXAI and the google-genai SDK's own
# GOOGLE_GENAI_USE_VERTEXAI, so a standard Vertex setup (which sets the latter)
# activates the Vertex path in our code too, rather than the two variables
# silently disagreeing. Either being truthy selects Vertex.
def _env_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("true", "1", "yes")


USE_VERTEXAI = _env_true("NCERT_USE_VERTEXAI") or _env_true("GOOGLE_GENAI_USE_VERTEXAI")

# GCP project + region, required for the Vertex AI backend. The Vertex backend
# will not activate on `vertexai=True` alone -- it also needs a project. We read
# the standard Google env vars so `gcloud` / ADC setups work with no extra
# configuration. GOOGLE_CLOUD_PROJECT is Google's own conventional variable.
GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("NCERT_GCP_PROJECT")
GCP_LOCATION = (
    os.environ.get("GOOGLE_CLOUD_LOCATION")
    or os.environ.get("NCERT_GCP_LOCATION")
    or "us-central1"
)


# --------------------------------------------------------------------------- #
# Retrieval / RAG
# --------------------------------------------------------------------------- #
RETRIEVER_K = int(os.environ.get("NCERT_RETRIEVER_K", 4))
CHAT_TEMPERATURE = float(os.environ.get("NCERT_CHAT_TEMPERATURE", 0.2))


@dataclass(frozen=True)
class BookRef:
    """A single book, identified by standard + subject. Used everywhere a
    function needs to know *which* book to operate on."""

    standard: str
    subject: str

    @property
    def slug(self) -> str:
        """Filesystem/collection-safe identifier, e.g. 'seven_maths'."""
        return f"{self.standard}_{self.subject}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.standard}/{self.subject}"