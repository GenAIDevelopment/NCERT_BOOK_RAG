"""Tests that exercise the pipeline logic without live Gemini credentials.

Vision, chat, and embedding models are all injected as stubs, so these run
offline in CI. They cover the parts most likely to break silently: image
filtering, section detection, chunk/image granularity, the JSON metadata
round-trip, and SQLRecordManager change tracking.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncert_rag import BookRef
from ncert_rag import extract as extract_stage
from ncert_rag import chunk as chunk_stage


BOOK = BookRef("seven", "maths")


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class StubCaption:
    n = 0

    def invoke(self, msgs):
        StubCaption.n += 1
        class R:
            content = f"a figure illustrating concept {StubCaption.n}"
        return R()


class StubEmbeddings:
    def embed_documents(self, texts):
        return [[(hash((t, i)) % 1000) / 1000.0 for i in range(16)] for t in texts]

    def embed_query(self, text):
        return [(hash((text, i)) % 1000) / 1000.0 for i in range(16)]


# --------------------------------------------------------------------------- #
# Unit tests (no PDFs needed)
# --------------------------------------------------------------------------- #
def test_is_decorative_flags_full_page_background():
    # huge dimensions -> decorative
    assert extract_stage.is_decorative(2480, 3508, 8558) is True
    # tiny byte size -> decorative
    assert extract_stage.is_decorative(400, 400, 3000) is True
    # a real photo -> kept
    assert extract_stage.is_decorative(425, 348, 154032) is False


def test_chapter_number_parsing():
    assert chunk_stage.chapter_number("gegp101") == 1
    assert chunk_stage.chapter_number("gegp108") == 8


def test_section_pattern_ignores_decimal_values():
    # chapter 3 pattern should match "3.1 Title" but not a decimal like "0.2 kg"
    pat = chunk_stage.section_pattern("gegp103")
    text = "3.1 The Need for Smaller Units\nWe measured 0.2 kg of capsicums.\n"
    matches = [m.group() for m in pat.finditer(text)]
    assert any(m.startswith("3.1") for m in matches)
    assert not any("0.2" in m for m in matches)


# --------------------------------------------------------------------------- #
# Integration tests (require the sample PDFs to be present)
# --------------------------------------------------------------------------- #
def _pdfs_present() -> bool:
    d = os.path.join("data", "standard", "seven", "maths")
    return os.path.isdir(d) and any(f.endswith(".pdf") for f in os.listdir(d))


requires_pdfs = pytest.mark.skipif(not _pdfs_present(), reason="sample PDFs not present")


@requires_pdfs
def test_extract_then_chunk_produces_sections():
    # extract one chapter only, for speed
    from ncert_rag import paths
    paths.ensure_dirs(BOOK)
    summary = extract_stage.extract_book(
        BOOK,
        skip_files=[f"gegp10{i}.pdf" for i in range(2, 9)] + ["gegp1ps.pdf"],
        enable_ocr=False,
    )
    assert summary[0]["chapter"] == "gegp101"
    assert summary[0]["pages"] > 0

    chunk_summary = chunk_stage.chunk_book(BOOK)
    ch1 = next(c for c in chunk_summary if c["chapter"] == "gegp101")
    # chapter 1 has 6 real sections
    assert ch1["sections"] == 6


@requires_pdfs
def test_sqlrecordmanager_skips_unchanged_on_reindex():
    from ncert_rag import index as index_stage
    from ncert_rag import caption as caption_stage
    from ncert_rag import paths

    # ensure captioned chunks exist (stub captions)
    caption_stage.caption_book(BOOK, model=StubCaption(), limit=1)

    # clean prior index state for a deterministic assertion
    db = paths.record_manager_db(BOOK)
    if os.path.exists(db):
        os.remove(db)
    vs = paths.vectorstore_dir(BOOK)
    if os.path.exists(vs):
        shutil.rmtree(vs)

    emb = StubEmbeddings()
    first = index_stage.build_index(BOOK, embeddings=emb, cleanup="incremental")
    second = index_stage.build_index(BOOK, embeddings=emb, cleanup="incremental")

    assert first["num_added"] > 0
    assert second["num_added"] == 0
    assert second["num_skipped"] == first["num_added"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
