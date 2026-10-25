"""Retrieval layer: finds ChromaDB stores, runs semantic search and formats the LLM context."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb

import config

# Characters of each retrieved chunk that are passed to the LLM
MAX_CHUNK_CHARS = 1000


def discover_chroma_backends(base_dir: str = ".") -> Dict[str, Dict]:
    """Find ChromaDB directories in base_dir and list the collections they contain.

    Directories that cannot be opened are still returned, with the error in the display name,
    so the UI can show why they are unusable.
    """
    backends = {}
    candidates = [p for p in Path(base_dir).iterdir() if p.is_dir() and "chroma" in p.name.lower()]

    for directory in candidates:
        try:
            collections = chromadb.PersistentClient(path=str(directory)).list_collections()
        except Exception as e:
            backends[f"{directory.name}::error"] = {
                "directory": str(directory),
                "collection_name": None,
                "display_name": f"{directory.name} (inaccessible: {str(e)[:50]})",
                "document_count": None,
            }
            continue

        for collection in collections:
            try:
                count = collection.count()
            except Exception:
                count = None
            count_label = f"{count} chunks" if count is not None else "count unavailable"
            backends[f"{directory.name}::{collection.name}"] = {
                "directory": str(directory),
                "collection_name": collection.name,
                "display_name": f"{directory.name} / {collection.name} ({count_label})",
                "document_count": count,
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str) -> Tuple[Optional["chromadb.Collection"], bool, str]:
    """Open an existing collection. Returns (collection, success, error_message)."""
    try:
        collection = chromadb.PersistentClient(path=chroma_dir).get_collection(name=collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(openai_api_key: str, collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Dict:
    """Embed the query with the ingestion model and return the closest chunks.

    mission_filter restricts results to one mission (e.g. "apollo_13"); None or "all" searches everything.
    """
    where = None
    if mission_filter and mission_filter.lower() != "all":
        where = {"mission_name": mission_filter}

    embedding = config.get_openai_client(openai_api_key).embeddings.create(
        model=config.EMBEDDING_MODEL, input=query
    ).data[0].embedding

    return collection.query(query_embeddings=[embedding], n_results=n_results, where=where)


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Render retrieved chunks with their mission, category and source for the system prompt."""
    if not documents:
        return ""

    parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        text = doc if len(doc) <= MAX_CHUNK_CHARS else doc[:MAX_CHUNK_CHARS] + "..."
        parts.append(
            f"Document {i}:\n"
            f"Mission: {_readable(meta.get('mission_name'))}\n"
            f"Category: {_readable(meta.get('document_category'))}\n"
            f"Source: {meta.get('source') or 'unknown'}\n"
            f"Text: {text}"
        )
    return "\n\n".join(parts)


def _readable(value: Optional[str]) -> str:
    """Turn a metadata slug like 'apollo_13' into 'Apollo 13'."""
    return (value or "unknown").replace("_", " ").title()
