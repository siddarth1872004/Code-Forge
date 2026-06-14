"""
Builds or rebuilds a Chroma collection from a local repository.

Usage (via CLI script):
    python scripts/index_repo.py /path/to/repo

Or programmatically:
    from src.rag.indexer import build_index
    build_index(repo_root="/path/to/repo", persist_dir=".chroma")
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.rag.chunker import Chunk, chunk_file, iter_repo_files

_UPSERT_BATCH = 100  # Chroma handles batches well up to a few hundred


def _get_collection(persist_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def _upsert_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    for i in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[i : i + _UPSERT_BATCH]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.content for c in batch],
            metadatas=[c.metadata for c in batch],
        )


def build_index(
    repo_root: str | Path,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    verbose: bool = False,
) -> chromadb.Collection:
    repo_root = Path(repo_root).resolve()
    persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
    collection_name = collection_name or repo_root.name

    collection = _get_collection(persist_dir, collection_name)

    all_chunks: list[Chunk] = []
    file_count = 0

    for abs_path in iter_repo_files(repo_root):
        file_chunks = chunk_file(repo_root, abs_path)
        if file_chunks:
            all_chunks.extend(file_chunks)
            file_count += 1
            if verbose:
                rel = abs_path.relative_to(repo_root)
                print(f"  indexed {rel} ({len(file_chunks)} chunk(s))")

    _upsert_chunks(collection, all_chunks)

    if verbose:
        print(f"\nIndexed {file_count} files → {len(all_chunks)} chunks into '{collection_name}'")

    return collection


def load_collection(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> chromadb.Collection:
    """Load an already-built collection without re-indexing."""
    repo_root_name = (
        collection_name
        or Path(os.environ.get("TARGET_REPO_PATH", ".")).resolve().name
    )
    persist_dir = persist_dir or os.environ.get("CHROMA_PERSIST_DIR", ".chroma")
    return _get_collection(persist_dir, repo_root_name)
