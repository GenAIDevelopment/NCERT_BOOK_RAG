"""Stage 3 - Captioning: image -> text, folded into chunk text.

Retrieval works over text/vectors; a raw image has no vector representation.
Captioning creates a *text description* of each figure and appends it to the
chunk the figure belongs to, so that when the chunk's text is embedded, the
embedding carries information about the image too - even though the embedding
model never sees a pixel.

Captions are cached per-chapter (captions/{chapter}.json: image_path ->
caption) so re-runs after adding a chapter don't re-pay for already-captioned
images.

Output: captioned_chunks/{chapter}.json  (chunks with captions in .text)
"""

from __future__ import annotations

import base64
import json
import os

from . import config
from .config import BookRef
from . import paths
from . import llm

CAPTION_PROMPT = (
    "This is a figure from a Class {standard} NCERT {subject} textbook. "
    "Describe what it shows in one or two sentences focused on the content a "
    "student would need. IMPORTANT: if the figure contains any numbers, labels, "
    "equations, or short text (e.g. values inside boxes, numbers on a number "
    "line, answers in a diagram), transcribe those exact values in your "
    "description -- in a maths book the figure often contains the actual answer "
    "(for example a box diagram might show '999', '+1', '1,000'). "
    "Example good caption: 'A number progression diagram showing the largest "
    "3-digit number 999, then +1 giving the smallest 4-digit number 1,000, "
    "continuing up to 1,00,000 (one lakh).' "
    "If the image is a plain decorative colour block with no informational "
    "content, reply exactly with: DECORATIVE. No preamble."
)


def caption_image(model, image_path: str, standard: str, subject: str) -> str:
    """Caption a single image with a vision model. Returns the caption text,
    or an empty string if the model flags it as purely decorative."""
    from langchain_core.messages import HumanMessage

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lstrip(".") or "png"

    prompt = CAPTION_PROMPT.format(standard=standard, subject=subject)
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/{ext};base64,{img_b64}"},
    ])
    caption = model.invoke([msg]).content.strip()
    if caption.upper().startswith("DECORATIVE"):
        return ""
    return caption


def caption_book(
    book: BookRef,
    model=None,
    limit: int | None = None,
    progress=None,
) -> list[dict]:
    """Caption every not-yet-cached image referenced by a book's chunks, then
    write captioned_chunks/{chapter}.json with captions folded into text.

    model:  inject a vision model (tests pass a stub); defaults to Gemini.
    limit:  cap on newly-captioned images per chapter, for a cheap dry run.
    """
    chunks_dir = paths.chunks_dir(book)
    captions_dir = paths.captions_dir(book)
    out_dir = paths.captioned_chunks_dir(book)
    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    model = model or llm.get_caption_model()

    chapter_files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".json"))
    summary = []

    for idx, fname in enumerate(chapter_files):
        if progress:
            progress(idx / max(len(chapter_files), 1), f"Captioning {fname}")

        chapter = fname.replace(".json", "")
        cache_path = os.path.join(captions_dir, fname)
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

        with open(os.path.join(chunks_dir, fname)) as f:
            chunks = json.load(f)

        all_images = sorted({img for c in chunks for img in c["image_paths"]})
        # only images not already cached, capped by limit if set
        pending = [img for img in all_images if img not in cache]
        if limit is not None:
            pending = pending[:limit]

        # Caption in parallel: these are I/O-bound API calls, so a thread pool
        # gives a large speedup over a sequential loop (which made a heavy
        # chapter like the decimals chapter, ~90 images, look frozen). Results
        # are collected then written once; per-image failures are isolated so
        # one bad image never halts the chapter.
        new_count = 0
        if pending:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _caption_one(path):
                return path, caption_image(model, path, book.standard, book.subject)

            with ThreadPoolExecutor(max_workers=config.CAPTION_CONCURRENCY) as pool:
                futures = {pool.submit(_caption_one, p): p for p in pending}
                for fut in as_completed(futures):
                    path = futures[fut]
                    try:
                        _, caption = fut.result()
                        cache[path] = caption
                        new_count += 1
                    except Exception as e:  # one bad image shouldn't halt a book
                        print(f"  ! failed to caption {path}: {e}")

        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

        # fold captions into chunk text (skip images flagged decorative -> "")
        for c in chunks:
            captions = [cache[p] for p in c["image_paths"] if cache.get(p)]
            if captions:
                c["text"] = c["text"] + "\n" + "\n".join(f"[Figure: {cap}]" for cap in captions)

        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(chunks, f, indent=2)

        summary.append({
            "chapter": chapter,
            "images": len(all_images),
            "newly_captioned": new_count,
        })

    if progress:
        progress(1.0, "Captioning complete")
    return summary