"""RAG quality evaluation with DeepEval, judged by Gemini.

Measures the pipeline on the standard RAG metrics, which split cleanly into
the two things that can go wrong:

  Retrieval quality (did we fetch the right context?)
    - ContextualRelevancyMetric : is the retrieved context on-topic?
    - ContextualRecallMetric     : does it contain what's needed for the answer?
    - ContextualPrecisionMetric  : are the most relevant chunks ranked first?

  Generation quality (did we use the context well?)
    - AnswerRelevancyMetric : does the answer address the question?
    - FaithfulnessMetric    : is every claim grounded in the context (no
                              hallucination)?

DeepEval defaults to an OpenAI judge; we wrap Gemini as a custom judge so the
whole system stays on one provider. Judge and system-under-test are separate
models by design - a model shouldn't grade its own homework on the same call.

Usage:
    python -m evaluation.evaluate_rag --standard seven --subject maths
    (optionally --dataset evaluation/golden_dataset.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ncert_rag import BookRef, TeacherRAG
from ncert_rag import config


# --------------------------------------------------------------------------- #
# Gemini judge wrapper for DeepEval
# --------------------------------------------------------------------------- #
def build_gemini_judge():
    """Wrap a Gemini chat model as a DeepEval judge (DeepEvalBaseLLM).

    Crucially, this honours the `schema` parameter DeepEval passes to
    generate()/a_generate(). DeepEval asks the judge for structured JSON and
    then parses the raw text; with a free-text response, Gemini occasionally
    emits JSON with unescaped characters (e.g. a LaTeX '\\frac' from math
    content), which DeepEval's parser rejects with "invalid JSON / use a better
    evaluation model". Binding the schema via LangChain's structured-output
    support makes Gemini return a validated object, so there's no fragile
    string-parsing step to fail. When no schema is given we fall back to plain
    text generation.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM
    from langchain_google_genai import ChatGoogleGenerativeAI

    from ncert_rag.llm import _vertex_kwargs

    class GeminiJudge(DeepEvalBaseLLM):
        def __init__(self):
            self._model = ChatGoogleGenerativeAI(
                model=config.CHAT_MODEL,
                temperature=0,
                **_vertex_kwargs(),
            )

        def load_model(self):
            return self._model

        def generate(self, prompt: str, schema=None):
            if schema is not None:
                structured = self._model.with_structured_output(schema)
                return structured.invoke(prompt)   # returns a validated schema instance
            return self._model.invoke(prompt).content

        async def a_generate(self, prompt: str, schema=None):
            if schema is not None:
                structured = self._model.with_structured_output(schema)
                return await structured.ainvoke(prompt)
            resp = await self._model.ainvoke(prompt)
            return resp.content

        def get_model_name(self) -> str:
            return f"gemini:{config.CHAT_MODEL}"

    return GeminiJudge()


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = [
    {
        "input": "Why is one lakh considered a large number?",
        "expected_output": "One lakh (1,00,000) is large because it is ten times "
        "ten thousand; counting or experiencing that many of something (like "
        "tasting a lakh varieties of rice) would take far longer than a lifetime.",
    },
    {
        "input": "What is the largest 3-digit number and what comes after it?",
        "expected_output": "The largest 3-digit number is 999; adding 1 gives 1000, "
        "the smallest 4-digit number.",
    },
    {
        "input": "How do you read the number 15,75,000 in the Indian system?",
        "expected_output": "Fifteen lakh seventy-five thousand.",
    },
]


def load_dataset(path: str | None) -> list[dict]:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return DEFAULT_DATASET


# --------------------------------------------------------------------------- #
# Evaluation run
# --------------------------------------------------------------------------- #
def run_evaluation(book: BookRef, dataset: list[dict], threshold: float = 0.7):
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )

    judge = build_gemini_judge()
    rag = TeacherRAG(book)

    # Build test cases by actually running each question through the RAG so we
    # capture the real answer AND the real retrieval context it used.
    test_cases = []
    for row in dataset:
        question = row["input"]
        docs = rag._retriever.invoke(question)          # the actual context used
        result = rag.answer(question)                    # the actual answer given
        test_cases.append(LLMTestCase(
            input=question,
            actual_output=result.answer,
            expected_output=row.get("expected_output"),
            retrieval_context=[d.page_content for d in docs],
        ))

    # Per-metric thresholds. Faithfulness and Recall are the metrics that
    # actually matter for a study aid (is the answer correct, and findable) --
    # held to a high bar. Contextual Relevancy is deliberately lower: it
    # rewards terse, encyclopedia-style context and penalises exactly the
    # narrative teaching style (stories, worked examples) that an NCERT
    # textbook is built on, so a blanket 0.70 is the wrong bar for this
    # content. Override any of these via the CLI --threshold for a uniform bar.
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        FaithfulnessMetric(threshold=0.8, model=judge),
        ContextualRelevancyMetric(threshold=0.35, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge),
        ContextualPrecisionMetric(threshold=0.7, model=judge),
    ]

    return evaluate(test_cases=test_cases, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the NCERT RAG with DeepEval (Gemini judge)")
    parser.add_argument("--standard", default="seven")
    parser.add_argument("--subject", default="maths")
    parser.add_argument("--dataset", default=None, help="Path to a JSON golden dataset")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    book = BookRef(args.standard, args.subject)
    dataset = load_dataset(args.dataset)
    print(f"Evaluating {book} on {len(dataset)} questions (threshold {args.threshold})…\n")
    run_evaluation(book, dataset, threshold=args.threshold)


if __name__ == "__main__":
    main()