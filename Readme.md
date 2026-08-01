# NCERT Book RAG

A **multimodal Retrieval-Augmented Generation** system over NCERT textbooks. It
extracts text *and* figures from the chapter PDFs, makes figures retrievable by
captioning them, indexes everything into a Chroma vector store with change
tracking, and answers student questions as a patient teacher — showing the
relevant diagrams alongside each explanation. Powered by Google Gemini.

Built and validated against **Class 7 Mathematics (Ganita Prakash)**, but the
folder convention generalises to any standard/subject.

---

## Why this exists (and what's non-obvious about it)

A naive "PDF → text → embed" pipeline throws away three things that matter in a
maths textbook:

1. **Figures.** Diagrams, number lines, and photos carry real content. But an
   image has no vector representation, so it's invisible to text search. We fix
   this by **captioning** each figure with a vision model and folding the
   caption into the surrounding chunk's text — so the embedding carries
   information about the image even though the embedder never sees a pixel.

2. **The right figures.** Textbook PDFs are full of decorative backgrounds, QR
   codes, and page borders. Extraction filters these out (by size/byte
   heuristics for backgrounds, by *decoding* for QR codes) so the index isn't
   polluted. It also recovers content that lives in three easily-missed places:
   **vector-drawn diagrams** (flowcharts built from PDF drawing operators, which
   `get_images()` never sees), and **text baked into image pixels** (recovered
   via confidence-filtered OCR).

3. **Structure.** Chunking splits on the book's own section numbers
   (`1.1`, `1.2`, …), not fixed windows, so a worked example is never severed
   from its heading — and each chunk carries only the figures from the page(s)
   its text actually spans.

---

## Pipeline

```
                  ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
   chapter PDFs → │  extract  │ → │  chunk   │ → │  caption  │ → │  index   │ → Chroma
                  └───────────┘   └──────────┘   └───────────┘   └──────────┘   + SQLRecordManager
                   text+figures    section-aware   image→text     embed & track
```

Each stage reads the previous stage's output directory and writes its own, so
each is independently re-runnable. **SQLRecordManager** means a re-index only
re-embeds chunks that actually changed — edit one chapter and re-run, and
everything else is skipped, not rebuilt.

---

## Install

```bash
# 1. system dependency for OCR
sudo apt-get install tesseract-ocr        # Debian/Ubuntu
# brew install tesseract                  # macOS

# 2. python deps
uv sync                                    # or: pip install -e .
```

### Configure Gemini

Two backends, chosen with one env var:

**Option A — Gemini Developer API (simplest, just an API key):**
```bash
export GOOGLE_API_KEY="your-key-from-ai-studio"
export NCERT_USE_VERTEXAI=false
```

**Option B — Vertex AI (GCP-billed):**
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
export NCERT_USE_VERTEXAI=true
```

See `.env.example` for all tunable settings.

---

## Data layout

Put chapter PDFs here (the folder convention *is* the config):

```
data/standard/{class}/{subject}/
    chapter1.pdf
    chapter2.pdf
    ...
```

Generated artefacts land in sibling folders (`extracted_text/`,
`extracted_images/`, `chunks/`, `captions/`, `captioned_chunks/`,
`vectorstore/`, `record_manager.sqlite`).

---

## Usage

### Streamlit UI (indexing + chat)

```bash
streamlit run streamlit_app/app.py
```

- **Index a Book** — pick a class/subject, choose cleanup mode, run the
  pipeline with live progress, and see the add/update/skip/delete report.
- **Ask a Teacher** — chat over a built index; answers cite sections and render
  the relevant figures inline.

### Command line

```bash
# full ingest
ncert-ingest --standard seven --subject maths

# cheap dry run: cap captions per chapter (eyeball quality before paying for all)
ncert-ingest --standard seven --subject maths --caption-limit 3

# full cleanup (also deletes vectors for removed chapters)
ncert-ingest --standard seven --subject maths --cleanup full
```

### As a library

```python
from ncert_rag import BookRef, ingest_book, TeacherRAG

book = BookRef("seven", "maths")
ingest_book(book)                       # extract → chunk → caption → index

rag = TeacherRAG(book)
result = rag.answer("Why is one lakh a large number?")
print(result.answer)                    # teacher-style explanation
print(result.images)                    # figure paths to display
print(result.sources)                   # [{chapter, section}, ...]
```

---

## Evaluation (DeepEval, Gemini judge)

Measures the pipeline on the standard RAG metrics, split by failure surface:

| Surface | Metric | Catches |
|---|---|---|
| Retrieval | Contextual Relevancy | off-topic retrieved chunks |
| Retrieval | Contextual Recall | missing information in context |
| Retrieval | Contextual Precision | relevant chunks ranked too low |
| Generation | Answer Relevancy | answers that dodge the question |
| Generation | Faithfulness | hallucinations not grounded in context |

```bash
python -m evaluation.evaluate_rag --standard seven --subject maths \
    --dataset evaluation/golden_dataset.json
```

The judge is a **separate** Gemini model from the one under test — a model
shouldn't grade its own output on the same call. Edit
`evaluation/golden_dataset.json` to add question/expected-answer pairs.

---

## Tests

```bash
pytest tests/ -v
```

Model calls are stubbed, so the suite runs offline. It covers image filtering,
section detection, the metadata round-trip, and SQLRecordManager change
tracking (a second identical index run must skip everything).

---

## Notes & caveats

- **PyMuPDF is AGPL-3.0.** Fine for internal/educational use; for a distributed
  commercial product, review licensing (Artifex sells a commercial licence; or
  `pypdf` is a permissive fallback with weaker image support).
- **`langchain-community` is being sunset.** `SQLRecordManager` still lives
  there for now; the import in `index.py` is isolated so a future move is a
  one-line change.
- **Vector-figure extraction is deliberately over-inclusive** — it captures
  candidate diagram regions and lets the captioning step (which flags purely
  decorative blocks) do the final filtering.
```
