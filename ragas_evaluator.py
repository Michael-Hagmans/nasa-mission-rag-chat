"""Response quality scoring with RAGAS (faithfulness and answer relevancy).

Run directly to score the whole pipeline on the test question set:
    python ragas_evaluator.py [questions.jsonl] [-n 5] [--max-chunk-chars 2500]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import config
import llm_client
import rag_client

try:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

METRIC_NAMES = ("faithfulness", "answer_relevancy")


def evaluate_response_quality(openai_key: str, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Score one answer against the chunks it was generated from.

    A metric is NaN when RAGAS cannot compute it, e.g. when the answer contains no factual statements.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=config.EVAL_MODEL, base_url=config.OPENAI_BASE_URL, api_key=openai_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OPENAI_BASE_URL, api_key=openai_key)
    )

    dataset = Dataset.from_dict({"question": [question], "contexts": [contexts], "answer": [answer]})
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    row = results.to_pandas().iloc[0]
    return {name: float(row[name]) for name in METRIC_NAMES}


def load_test_questions(path: Path) -> List[Dict]:
    """Read a JSON Lines file with one {"question": ...} object per line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def batch_evaluation(openai_key: str, questions_path: Path, collection,
                     n_results: int = config.DEFAULT_N_RESULTS) -> Dict[str, float]:
    """Run each test question through retrieval and generation and average the RAGAS scores.

    NaN scores are left out of the averages.
    """
    scores = {name: [] for name in METRIC_NAMES}

    for entry in load_test_questions(questions_path):
        question = entry["question"]
        results = rag_client.retrieve_documents(openai_key, collection, question, n_results)
        documents, metadatas = results["documents"][0], results["metadatas"][0]
        answer = llm_client.generate_response(
            openai_key, question, rag_client.format_context(documents, metadatas), []
        )

        result = evaluate_response_quality(openai_key, question, answer, documents)
        for name in METRIC_NAMES:
            value = result.get(name)
            if isinstance(value, float) and not math.isnan(value):
                scores[name].append(value)

    return {name: sum(values) / len(values) if values else float("nan") for name, values in scores.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the RAG pipeline on a set of test questions")
    parser.add_argument("questions_file", nargs="?", default="test_questions.json",
                        help="JSON Lines file with one {\"question\": ...} object per line")
    parser.add_argument("-n", "--n-results", type=int, default=config.DEFAULT_N_RESULTS,
                        help="Chunks retrieved per question")
    parser.add_argument("--max-chunk-chars", type=int, default=rag_client.MAX_CHUNK_CHARS,
                        help="Characters of each chunk passed to the model")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")

    collection, ok, error = rag_client.initialize_rag_system(config.CHROMA_DIR, config.COLLECTION_NAME)
    if not ok:
        sys.exit(f"Could not open collection: {error}")

    rag_client.MAX_CHUNK_CHARS = args.max_chunk_chars
    averages = batch_evaluation(api_key, Path(args.questions_file), collection, args.n_results)

    print(f"n_results={args.n_results}  max_chunk_chars={args.max_chunk_chars}")
    for name, value in averages.items():
        print(f"{name}: {value:.3f}")
