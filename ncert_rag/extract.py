"""Stage 1 - Extraction: PDF -> per-page text + figures.

Pulls, for every page of every chapter PDF:
  * the page's text (via PyMuPDF's native text layer);
  * embedded raster figures, minus decorative backgrounds and QR codes;
  * vector-drawn figures (flowcharts / box diagrams) that are NOT embedded
    images and so are invisible to get_images();
  * text baked into image pixels, recovered via confidence-filtered OCR and
    folded into the page's text so it becomes searchable downstream.

Output: extracted_text/{chapter}.json  (list of per-page dicts)
        extracted_images/{chapter}_p{n}_{idx}.{ext}  (kept figures)

Why PyMuPDF: it exposes both the text layer and raw per-image bytes/metadata
from the same Page object, plus page rasterisation (get_pixmap) for the
vector-figure and verification paths. Note: PyMuPDF is AGPL-3.0; for a
distributed/commercial product, review licensing (a commercial licence is
available from Artifex, or pypdf is a permissive-but-less-capable fallback).
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass, field

import fitz  # PyMuPDF

from . import config
from .config import BookRef
from . import paths


@dataclass
class ExtractedPage:
    """One page's worth of extracted content."""

    chapter: str
    page_number: int
    text: str
    image_paths: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-image classification helpers
# --------------------------------------------------------------------------- #
def is_decorative(width: int, height: int, byte_len: int) -> bool:
    """True for full-page background tints and simple decorative graphics.

    Two independent signals, either sufficient:
      * huge dimensions (a page-sized background), OR
      * tiny byte size (a flat colour / simple texture compresses to almost
        nothing, regardless of pixel dimensions).
    """
    if width > config.DECORATIVE_MIN_DIM and height > config.DECORATIVE_MIN_DIM:
        return True
    if byte_len < config.DECORATIVE_MAX_BYTES:
        return True
    return False


def is_qr_code(image_bytes: bytes) -> bool:
    """True if the image is a decodable QR code.

    This book puts ePathshala QR links on chapter/section pages; they carry
    no content for a RAG index. Detecting by *decoding* (cv2.QRCodeDetector)
    rather than by a "small square" heuristic avoids false-positive removal
    of genuine square diagrams.
    """
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    try:
        _, points, _ = cv2.QRCodeDetector().detectAndDecode(img)
    except Exception:
        return False
    return points is not None


def ocr_image_text(image_bytes: bytes) -> str:
    """Recover real text baked into an image's pixels (invisible to the PDF
    text layer), keeping only high-confidence words.

    Confidence filtering is essential: raw OCR on decorative/textured images
    yields garbage ("tHtttes FEELS"), which would pollute the index. Words
    below OCR_MIN_CONFIDENCE are dropped; results shorter than OCR_MIN_CHARS
    are discarded entirely.
    """
    import pytesseract
    from pytesseract import Output
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception:
        return ""

    words = [
        w for w, c in zip(data["text"], data["conf"])
        if w.strip() and int(c) >= config.OCR_MIN_CONFIDENCE
    ]
    text = " ".join(words)
    return text if len(text) >= config.OCR_MIN_CHARS else ""


def extract_vector_figure(page, chapter: str, page_index: int, image_out_dir: str) -> str | None:
    """Render the bounding region of vector-drawn shapes on a page, if any.

    Flowcharts and box-and-arrow diagrams are drawn with PDF vector operators,
    not embedded as raster images, so get_images() never sees them. We union
    the bounding boxes of shapes whose individual area is between
    VECTOR_FIG_MIN/MAX_AREA_FRAC of the page (excluding tiny underlines and
    near-full-page backgrounds) and rasterise that region.

    We deliberately can't tell geometry-only whether a big coloured region is
    a real diagram or a decorative highlight box; the downstream captioning
    step (a vision model) makes that judgement.
    """
    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    candidates = [
        d["rect"] for d in page.get_drawings()
        if config.VECTOR_FIG_MIN_AREA_FRAC
        < (d["rect"].width * d["rect"].height) / page_area
        < config.VECTOR_FIG_MAX_AREA_FRAC
    ]
    if not candidates:
        return None

    x0 = min(r.x0 for r in candidates)
    y0 = min(r.y0 for r in candidates)
    x1 = max(r.x1 for r in candidates)
    y1 = max(r.y1 for r in candidates)

    img_path = os.path.join(image_out_dir, f"{chapter}_p{page_index}_vec.png")
    page.get_pixmap(dpi=config.VECTOR_FIG_DPI, clip=fitz.Rect(x0, y0, x1, y1)).save(img_path)
    return img_path


# --------------------------------------------------------------------------- #
# Per-PDF and per-book extraction
# --------------------------------------------------------------------------- #
def _strip_boilerplate_line(line: str) -> bool:
    """True if a single line is boilerplate that should be dropped."""
    import re
    s = line.strip()
    if not s:
        return False
    if s in config.BOILERPLATE_EXACT:
        return True
    for pat in config.BOILERPLATE_PATTERNS:
        if re.match(pat, s):
            return True
    return False


def _detect_running_headers(pages_text: list[str], min_fraction: float = 0.5) -> set[str]:
    """Find lines that repeat across a large fraction of a chapter's pages --
    these are running headers/footers (e.g. the chapter-title header at the
    top of most pages), which vary per chapter and so can't be hardcoded.

    Conservative on purpose: only lines longer than 8 characters that appear on
    at least half the pages qualify. This avoids stripping short content
    fragments that legitimately recur (a value like '999', a label repeated
    across worked examples, a heading), which an earlier, looser version could
    have removed -- costing retrieval recall on figure-adjacent content."""
    from collections import Counter

    counts: Counter = Counter()
    for text in pages_text:
        seen_on_page = {ln.strip() for ln in text.split("\n") if len(ln.strip()) > 8}
        for ln in seen_on_page:
            counts[ln] += 1

    threshold = max(4, int(len(pages_text) * min_fraction))
    return {ln for ln, c in counts.items() if c >= threshold}


def _clean_page_text(text: str, running_headers: set[str]) -> str:
    """Drop boilerplate lines (fixed patterns + detected running headers)."""
    kept = []
    for line in text.split("\n"):
        if _strip_boilerplate_line(line):
            continue
        if line.strip() in running_headers:
            continue
        kept.append(line)
    return "\n".join(kept)


def extract_pdf(pdf_path: str, image_out_dir: str, enable_ocr: bool = True) -> list[ExtractedPage]:
    """Extract every page of one chapter PDF."""
    chapter = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(image_out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages: list[ExtractedPage] = []

    for i, page in enumerate(doc):
        text = page.get_text()
        kept_images: list[str] = []
        ocr_snippets: list[str] = []

        for img_idx, im in enumerate(page.get_images(full=True)):
            base = doc.extract_image(im[0])
            w, h, data, ext = base["width"], base["height"], base["image"], base["ext"]

            if is_decorative(w, h, len(data)):
                continue
            if len(data) < config.MIN_CONTENT_BYTES:
                continue
            if is_qr_code(data):
                continue

            img_path = os.path.join(image_out_dir, f"{chapter}_p{i}_{img_idx}.{ext}")
            with open(img_path, "wb") as f:
                f.write(data)
            kept_images.append(img_path)

            if enable_ocr:
                ocr_text = ocr_image_text(data)
                if ocr_text:
                    ocr_snippets.append(ocr_text)

        vector_fig = extract_vector_figure(page, chapter, i, image_out_dir)
        if vector_fig:
            kept_images.append(vector_fig)

        if ocr_snippets:
            text = text + "\n" + "\n".join(f"[Image text: {s}]" for s in ocr_snippets)

        pages.append(ExtractedPage(chapter=chapter, page_number=i, text=text, image_paths=kept_images))

    doc.close()

    # Two-pass boilerplate removal: now that we have every page's text, detect
    # running headers/footers that repeat across the chapter and strip them
    # (plus the fixed publication-footer/page-number patterns) from each page.
    # This removes the "Reprint 2026-27", running chapter-title header, and
    # bare-page-number noise that was diluting Contextual Relevancy without
    # touching real content.
    running_headers = _detect_running_headers([p.text for p in pages])
    for p in pages:
        p.text = _clean_page_text(p.text, running_headers)

    return pages


def extract_book(
    book: BookRef,
    skip_files: list[str] | None = None,
    enable_ocr: bool = True,
    progress=None,
) -> list[dict]:
    """Extract every chapter PDF for a book, writing per-chapter JSON.

    progress: optional callable(fraction: float, message: str) for UI feedback.
    Returns a per-chapter summary list.
    """
    skip_files = skip_files or []
    source_dir = paths.source_pdf_dir(book)
    text_out_dir = paths.extracted_text_dir(book)
    image_out_dir = paths.extracted_images_dir(book)
    os.makedirs(text_out_dir, exist_ok=True)

    pdf_files = sorted(
        f for f in os.listdir(source_dir)
        if f.endswith(".pdf") and f not in skip_files
    )

    summary = []
    for idx, fname in enumerate(pdf_files):
        if progress:
            progress(idx / max(len(pdf_files), 1), f"Extracting {fname}")

        pages = extract_pdf(os.path.join(source_dir, fname), image_out_dir, enable_ocr=enable_ocr)
        chapter = pages[0].chapter

        out_json = os.path.join(text_out_dir, f"{chapter}.json")
        with open(out_json, "w") as f:
            json.dump([asdict(p) for p in pages], f, indent=2)

        summary.append({
            "chapter": chapter,
            "pages": len(pages),
            "total_chars": sum(len(p.text) for p in pages),
            "kept_images": sum(len(p.image_paths) for p in pages),
        })

    with open(os.path.join(text_out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if progress:
        progress(1.0, "Extraction complete")
    return summary