#!/usr/bin/env python3
"""
Build (or rebuild) the Chroma vector index for a target repository.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/index_repo.py /path/to/repo
    PYTHONPATH=. .venv/bin/python scripts/index_repo.py /path/to/repo --persist-dir .chroma
    PYTHONPATH=. .venv/bin/python scripts/index_repo.py /path/to/repo --collection my-repo
"""

import argparse
import sys
from pathlib import Path

# Allow running from any working directory as long as PYTHONPATH=. is set.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a repository into Chroma for RAG.")
    parser.add_argument("repo", help="Path to the target repository root.")
    parser.add_argument(
        "--persist-dir",
        default=".chroma",
        help="Directory to persist the Chroma index (default: .chroma).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name (default: repo directory name).",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"Error: {repo} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Indexing {repo} → {args.persist_dir} ...")
    build_index(
        repo_root=repo,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        verbose=True,
    )
    print("\nDone. Set these env vars before running the pipeline:")
    print(f"  TARGET_REPO_PATH={repo}")
    print(f"  CHROMA_PERSIST_DIR={Path(args.persist_dir).resolve()}")


if __name__ == "__main__":
    main()
