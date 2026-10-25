"""Show which chunks the retriever returns for a query, with similarity scores and metadata.

Usage:
    python inspect_retrieval.py "Why did the oxygen tank fail on Apollo 13?" -n 5
    python inspect_retrieval.py "Who flew Apollo 11?" --mission apollo_11 --show-context
"""

import argparse
import os
import sys

import config
import rag_client


def main():
    parser = argparse.ArgumentParser(description="Inspect retrieval results for a query")
    parser.add_argument("query", nargs="?", default="What went wrong on Apollo 13?")
    parser.add_argument("-n", "--n-results", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--mission", help="Restrict the search to one mission, e.g. apollo_13")
    parser.add_argument("--show-context", action="store_true",
                        help="Also print the context exactly as the LLM receives it")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY first.")

    collection, ok, error = rag_client.initialize_rag_system(config.CHROMA_DIR, config.COLLECTION_NAME)
    if not ok:
        sys.exit(f"Could not open collection: {error}")
    print(f"Collection '{config.COLLECTION_NAME}' has {collection.count()} chunks\n")

    results = rag_client.retrieve_documents(api_key, collection, args.query, args.n_results, args.mission)
    documents, metadatas = results["documents"][0], results["metadatas"][0]

    print(f"QUERY: {args.query}\n")
    for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, results["distances"][0]), start=1):
        similarity = 1 - dist / 2  # OpenAI embeddings are unit length, so this is cosine similarity
        print("=" * 80)
        print(f"#{rank}  similarity={similarity:.3f}  distance={dist:.3f}")
        print(f"mission={meta.get('mission_name')}  source={meta.get('source')}  "
              f"category={meta.get('document_category')}  chunk={meta.get('chunk_index')}")
        print("-" * 80)
        print(doc[:800] + ("..." if len(doc) > 800 else ""))
    print("=" * 80)

    if args.show_context:
        print("\nCONTEXT SENT TO THE LLM:\n")
        print(rag_client.format_context(documents, metadatas))


if __name__ == "__main__":
    main()
