"""Stage 2 - Chunking: per-page text -> section-aware chunks.

NCERT chapters number their own sections ("1.1", "1.2", ...). We split on
those headers rather than on fixed character windows, so a worked example is
never severed from its heading. A section longer than MAX_CHUNK_CHARS is then
size-split as a fallback.

Each resulting chunk carries only the images from the page(s) its own text
actually spans (not the whole section's images), so retrieval surfaces the
right figure rather than every figure in a multi-page section.

Output: chunks/{chapter}.json  (list of chunk dicts)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .config import BookRef
from . import paths


@dataclass
class Chunk:
    chapter: str
    section: str
    pages: list = field(default_factory=list)
    image_paths: list = field(default_factory=list)
    text: str = ""


def chapter_number(chapter: str) -> int:
    """'gegp103' -> 3. Anchoring the section regex to the real chapter number
    is what prevents decimal values in the body ("0.2 kg", "6.4 6.45") from
    being mistaken for section headers - a real problem in the decimals
    chapter."""
    digits = re.findall(r"\d", chapter)
    return int(digits[-1]) if digits else 0


def section_pattern(chapter: str) -> re.Pattern:
    """Regex for THIS chapter's section headers.

    Requires the chapter number, then optionally a couple of stray
    formatting/control characters (a few headers carry a bullet glyph like
    \\x07 before the title), then a letter - not another digit.
    """
    n = chapter_number(chapter)
    return re.compile(rf"^{n}\.\d+\s+[^\w\s]{{0,2}}[A-Za-z].+$", re.MULTILINE)


def chunk_chapter(pages: list[dict]) -> list[Chunk]:
    """Turn one chapter's per-page JSON into section-aware chunks."""
    chapter = pages[0]["chapter"]
    pattern = section_pattern(chapter)

    # Flatten pages into one string, remembering where each page starts so we
    # can map any character position back to a page number afterwards.
    full_text = ""
    page_offsets: list[tuple[int, int, list]] = []
    for p in pages:
        page_offsets.append((len(full_text), p["page_number"], p["image_paths"]))
        full_text += p["text"] + "\n"

    matches = list(pattern.finditer(full_text))
    boundaries = [m.start() for m in matches] + [len(full_text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_CHARS, chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks: list[Chunk] = []
    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        section_text = full_text[start:end].strip()
        if not section_text:
            continue
        section_title = section_text.split("\n")[0].strip()

        # page offsets re-based to be relative to this section's text
        section_pages = [
            (offset - start, pnum, imgs)
            for offset, pnum, imgs in page_offsets
            if start <= offset < end
        ]

        search_cursor = 0
        for sub in splitter.split_text(section_text):
            find_from = max(0, search_cursor - config.CHUNK_OVERLAP)
            sub_start = section_text.find(sub, find_from)
            if sub_start == -1:
                sub_start = search_cursor
            sub_end = sub_start + len(sub)
            search_cursor = sub_end

            images_in_range = [
                img for offset, _, imgs in section_pages
                if sub_start <= offset < sub_end for img in imgs
            ]
            pages_in_range = sorted({
                pnum for offset, pnum, _ in section_pages
                if sub_start <= offset < sub_end
            })
            # a sub-chunk starting mid-page still belongs to that page
            if not pages_in_range:
                preceding = [p for p in section_pages if p[0] <= sub_start]
                if preceding:
                    _, pnum, imgs = preceding[-1]
                    pages_in_range = [pnum]
                    images_in_range = list(imgs)

            chunks.append(Chunk(
                chapter=chapter,
                section=section_title,
                pages=pages_in_range,
                image_paths=images_in_range,
                text=sub,
            ))

    return chunks


def chunk_book(book: BookRef, progress=None) -> list[dict]:
    """Chunk every chapter of a book, writing chunks/{chapter}.json."""
    text_dir = paths.extracted_text_dir(book)
    out_dir = paths.chunks_dir(book)
    os.makedirs(out_dir, exist_ok=True)

    chapter_files = sorted(
        f for f in os.listdir(text_dir)
        if f.endswith(".json") and not f.startswith("_")
    )

    summary = []
    for idx, fname in enumerate(chapter_files):
        if progress:
            progress(idx / max(len(chapter_files), 1), f"Chunking {fname}")

        with open(os.path.join(text_dir, fname)) as f:
            pages = json.load(f)

        chunks = chunk_chapter(pages)
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2)

        summary.append({
            "chapter": pages[0]["chapter"],
            "sections": len({c.section for c in chunks}),
            "chunks": len(chunks),
        })

    if progress:
        progress(1.0, "Chunking complete")
    return summary
