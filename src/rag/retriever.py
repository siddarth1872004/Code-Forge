"""
Semantic search over a Chroma collection.
Results are compressed before being returned to fit within the generator's context budget.
"""

from __future__ import annotations

import chromadb

from src.utils.compress import strip_python, token_budget

TOP_K = 3          # reduced from 5; 3 high-relevance chunks beat 5 diluted ones
_CHUNK_BUDGET = 250  # tokens per chunk after compression


def search(collection: chromadb.Collection, query: str, k: int = TOP_K) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs: list[str] = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]

    if not docs:
        return "No relevant code found."

    parts: list[str] = [f"Found {len(docs)} chunk(s):\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        file_ref = meta.get("file", "?")
        start    = meta.get("start_line", "?")
        end      = meta.get("end_line", "?")
        symbol   = meta.get("symbol", "")
        relevance = f"{(1 - dist) * 100:.0f}%"

        header = f"[{i}] {file_ref}:{start}-{end}"
        if symbol and symbol != "<module>":
            header += f" ({symbol})"
        header += f" {relevance}"

        # Compress Python chunks before adding to context
        if file_ref.endswith(".py"):
            doc = strip_python(doc)
        doc = token_budget(doc, _CHUNK_BUDGET)

        parts.append(f"--- {header} ---")
        parts.append(doc)
        parts.append("---")

    return "\n".join(parts)
