#!/usr/bin/env python3
"""
Embedding pipeline for NASA mission text files.

Reads the extracted mission texts (Apollo 11, Apollo 13, Challenger), splits them into
overlapping token chunks, embeds each chunk with OpenAI and stores the result in a
persistent ChromaDB collection used by the chat app.

Only .txt files are processed so JSON copies of the same sources are not indexed twice.
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import tiktoken

import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Chunk = Tuple[str, Dict[str, Any]]

DATA_DIRS = ['apollo11', 'apollo13', 'challenger']


class ChromaEmbeddingPipelineTextOnly:
    """Builds and maintains a ChromaDB collection of OpenAI-embedded text chunks."""

    def __init__(self,
                 openai_api_key: str,
                 chroma_persist_directory: str = config.CHROMA_DIR,
                 collection_name: str = config.COLLECTION_NAME,
                 embedding_model: str = config.EMBEDDING_MODEL,
                 chunk_size: int = 500,
                 chunk_overlap: int = 100):
        """
        Args:
            openai_api_key: OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name: Name of the ChromaDB collection
            embedding_model: OpenAI embedding model to use
            chunk_size: Maximum chunk length in tokens
            chunk_overlap: Tokens shared by consecutive chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.client = config.get_openai_client(openai_api_key)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Same tokenizer as text-embedding-3-small
        self.encoding = tiktoken.get_encoding("cl100k_base")

        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into overlapping token windows.

        Each chunk gets its own copy of the metadata with a 1-based chunk_index.
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_size:
            return [(text, {**metadata, "chunk_index": 1})]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for chunk_index, start in enumerate(range(0, len(tokens), step), start=1):
            window = tokens[start:start + self.chunk_size]
            chunks.append((self.encoding.decode(window), {**metadata, "chunk_index": chunk_index}))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """Delete all chunks whose source name contains source_pattern. Returns the number deleted."""
        try:
            all_docs = self.collection.get(include=["metadatas"])
            ids_to_delete = [
                doc_id for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas'])
                if source_pattern in metadata.get('source', '')
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} documents matching source pattern: {source_pattern}")
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_document_ids(self, file_path: Path) -> List[str]:
        """Return the IDs of all chunks that were created from file_path."""
        result = self.collection.get(
            where={"$and": [
                {"source": file_path.stem},
                {"mission_name": self.extract_mission_from_path(file_path)},
            ]},
            include=[],
        )
        return result["ids"]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed several texts with a single API call."""
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    @staticmethod
    def generate_document_id(metadata: Dict[str, Any]) -> str:
        """Stable chunk ID (mission_source_chunk_0001), so re-runs update instead of duplicating."""
        return f"{metadata['mission_name']}_{metadata['source']}_chunk_{metadata['chunk_index']:04d}"

    def process_text_file(self, file_path: Path) -> List[Chunk]:
        """Read a text file and return its chunks with metadata."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return []

        if not content.strip():
            return []

        metadata = {
            'source': file_path.stem,
            'file_path': str(file_path),
            'file_type': 'text',
            'content_type': 'full_text',
            'mission_name': self.extract_mission_from_path(file_path),
            'data_type': self.extract_data_type_from_path(file_path),
            'document_category': self.extract_document_category_from_filename(file_path.name),
            'file_size': len(content),
            'processed_timestamp': datetime.now().isoformat()
        }
        return self.chunk_text(content, metadata)

    @staticmethod
    def extract_mission_from_path(file_path: Path) -> str:
        """Extract mission name from file path"""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        else:
            return 'unknown'

    @staticmethod
    def extract_data_type_from_path(file_path: Path) -> str:
        """Extract data type from file path"""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        else:
            return 'document'

    @staticmethod
    def extract_document_category_from_filename(filename: str) -> str:
        """Extract document category from filename"""
        filename_lower = filename.lower()

        # Apollo transcript types
        if 'pao' in filename_lower:
            return 'public_affairs_officer'
        elif 'cm' in filename_lower:
            return 'command_module'
        elif 'tec' in filename_lower:
            return 'technical'
        elif 'flight_plan' in filename_lower:
            return 'flight_plan'

        # Challenger audio segments
        elif 'mission_audio' in filename_lower:
            return 'mission_audio'

        # NASA archive documents
        elif 'ntrs' in filename_lower:
            return 'nasa_archive'
        elif '19900066485' in filename_lower:
            return 'technical_report'
        elif '19710015566' in filename_lower:
            return 'mission_report'

        elif 'full_text' in filename_lower:
            return 'complete_document'
        else:
            return 'general_document'

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """Collect the .txt files in the mission folders below base_path, skipping hidden files and summaries."""
        base_path = Path(base_path)
        files_to_process = []

        for data_dir in DATA_DIRS:
            dir_path = base_path / data_dir
            if not dir_path.exists():
                logger.warning(f"Data directory not found: {dir_path}")
                continue
            text_files = [
                f for f in dir_path.glob('**/*.txt')
                if not f.name.startswith('.') and 'summary' not in f.name.lower()
            ]
            files_to_process.extend(text_files)
            logger.info(f"Found {len(text_files)} text files in {dir_path}")

        logger.info(f"Total text files to process: {len(files_to_process)}")
        return files_to_process

    def add_documents_to_collection(self, documents: List[Chunk], file_path: Path,
                                    batch_size: int = 50, update_mode: str = 'skip') -> Dict[str, int]:
        """
        Embed and store chunks in batches.

        Args:
            documents: List of (text, metadata) tuples
            file_path: Path to the source file
            batch_size: Number of chunks embedded per API call
            update_mode: How to handle chunks that already exist:
                        'skip' - leave existing chunks unchanged
                        'update' - re-embed and overwrite existing chunks
                        'replace' - delete all chunks of the file first, then add

        Returns:
            Counts of added, updated, skipped and failed chunks
        """
        stats = {'added': 0, 'updated': 0, 'skipped': 0, 'failed': 0}
        if not documents:
            return stats

        if update_mode == 'replace':
            existing_ids = self.get_file_document_ids(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)

        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            ids = [self.generate_document_id(metadata) for _, metadata in batch]
            existing = set(self.collection.get(ids=ids, include=[])["ids"])

            to_add, to_update = [], []
            for doc_id, (text, metadata) in zip(ids, batch):
                if doc_id not in existing:
                    to_add.append((doc_id, text, metadata))
                elif update_mode == 'update':
                    to_update.append((doc_id, text, metadata))
                else:
                    stats['skipped'] += 1

            for items, write, stat_key in ((to_add, self.collection.add, 'added'),
                                           (to_update, self.collection.update, 'updated')):
                if not items:
                    continue
                item_ids, texts, metadatas = (list(column) for column in zip(*items))
                try:
                    write(ids=item_ids, documents=texts, metadatas=metadatas,
                          embeddings=self.get_embeddings(texts))
                    stats[stat_key] += len(items)
                except Exception as e:
                    logger.error(f"Error writing {len(items)} chunks from {file_path}: {e}")
                    stats['failed'] += len(items)

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = 'skip',
                              batch_size: int = 50) -> Dict[str, Any]:
        """Chunk, embed and store every mission text file below base_path. Returns run statistics."""
        stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'documents_failed': 0,
            'errors': 0,
            'missions': {}
        }

        for file_path in self.scan_text_files_only(base_path):
            logger.info(f"Processing {file_path}")
            try:
                chunks = self.process_text_file(file_path)
                result = self.add_documents_to_collection(chunks, file_path, batch_size, update_mode)
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"Error processing {file_path}: {e}")
                continue

            mission_stats = stats['missions'].setdefault(
                self.extract_mission_from_path(file_path),
                {'files': 0, 'chunks': 0, 'added': 0, 'updated': 0, 'skipped': 0}
            )
            mission_stats['files'] += 1
            mission_stats['chunks'] += len(chunks)
            for key in ('added', 'updated', 'skipped'):
                mission_stats[key] += result[key]
                stats[f'documents_{key}'] += result[key]
            stats['documents_failed'] += result['failed']
            stats['total_chunks'] += len(chunks)
            stats['files_processed'] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get name, size and metadata of the collection"""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Run a similarity search, embedding the query with the ingestion model."""
        embedding = self.get_embeddings([query_text])[0]
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Count chunks per mission, data type, document category and file type"""
        try:
            all_docs = self.collection.get(include=["metadatas"])
            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            stats = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }
            for metadata in all_docs['metadatas']:
                for stat_key, meta_key in (('missions', 'mission_name'),
                                           ('data_types', 'data_type'),
                                           ('document_categories', 'document_category'),
                                           ('file_types', 'file_type')):
                    value = metadata.get(meta_key, 'unknown')
                    stats[stat_key][value] = stats[stat_key].get(value, 0) + 1
            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default=config.DATA_DIR, help='Path to data directories')
    parser.add_argument('--openai-key', default=os.getenv('OPENAI_API_KEY'),
                        help='OpenAI API key (default: OPENAI_API_KEY environment variable)')
    parser.add_argument('--chroma-dir', default=config.CHROMA_DIR, help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default=config.COLLECTION_NAME, help='Collection name')
    parser.add_argument('--embedding-model', default=config.EMBEDDING_MODEL, help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size in tokens')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap in tokens')
    parser.add_argument('--batch-size', type=int, default=50, help='Chunks embedded per API call')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                        help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics')
    parser.add_argument('--delete-source', help='Delete all documents from a specific source pattern')

    args = parser.parse_args()
    if not args.openai_key:
        parser.error("Provide --openai-key or set OPENAI_API_KEY")

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    if args.delete_source:
        pipeline.delete_documents_by_source(args.delete_source)
        return

    if args.stats_only:
        logger.info("Collection Statistics:")
        for key, value in pipeline.get_collection_stats().items():
            logger.info(f"{key}: {value}")
        return

    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode,
                                           batch_size=args.batch_size)
    processing_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Documents failed: {stats['documents_failed']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    logger.info("Mission breakdown:")
    for mission, mission_stats in stats['missions'].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, "
                    f"Skipped: {mission_stats['skipped']}")

    collection_info = pipeline.get_collection_info()
    logger.info(f"Collection: {collection_info['collection_name']}")
    logger.info(f"Total documents in collection: {collection_info['document_count']}")

    if args.test_query:
        logger.info(f"Testing query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        logger.info(f"Found {len(results['documents'][0])} results:")
        for i, doc in enumerate(results['documents'][0][:3], start=1):
            logger.info(f"Result {i}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
