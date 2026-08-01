"""Command-line entrypoint for running the ingestion pipeline.

    ncert-ingest --standard seven --subject maths
    ncert-ingest --standard seven --subject maths --caption-limit 3   # dry run
    ncert-ingest --standard seven --subject maths --cleanup full
"""

from __future__ import annotations

import argparse

from .config import BookRef
from . import pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the NCERT RAG ingestion pipeline")
    parser.add_argument("--standard", default="seven")
    parser.add_argument("--subject", default="maths")
    parser.add_argument("--skip", nargs="*", default=["gegp1ps.pdf"],
                        help="PDF filenames to skip (e.g. answer keys)")
    parser.add_argument("--caption-limit", type=int, default=None,
                        help="Cap new captions per chapter (cheap dry run)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable image OCR")
    parser.add_argument("--cleanup", default="incremental",
                        choices=["incremental", "full", "none"])
    args = parser.parse_args()

    book = BookRef(args.standard, args.subject)

    def show(frac, msg):
        print(f"[{frac*100:5.1f}%] {msg}")

    result = pipeline.ingest_book(
        book,
        skip_files=args.skip,
        caption_limit=args.caption_limit,
        enable_ocr=not args.no_ocr,
        cleanup=(None if args.cleanup == "none" else args.cleanup),
        progress=show,
    )

    print("\nIndex change report:", result["index"])


if __name__ == "__main__":
    main()
