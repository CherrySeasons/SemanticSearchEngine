"""
Phase 4 - Labeling helper
Hand-labeling eval/queries.json requires knowing which chunk_ids are
relevant to a given query. This script lets you search metadata.jsonl by
keyword so you can find candidate chunk_ids without manually scrolling
through the file.

Usage:
    python src/evaluation/find_chunks.py "photosynthesis"
    python src/evaluation/find_chunks.py "capital of France" --limit 10
"""
import argparse
import json
from pathlib import Path

METADATA_PATH = Path("data/processed/metadata.jsonl")


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword", help="Keyword/phrase to search for (case-insensitive)")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    records = load_metadata(METADATA_PATH)
    keyword = args.keyword.lower()

    matches = [
        r for r in records
        if keyword in r["title"].lower() or keyword in r["text"].lower()
    ]

    print(f"Found {len(matches)} matches for '{args.keyword}' (showing up to {args.limit})\n")
    for r in matches[: args.limit]:
        print(f"chunk_id: {r['chunk_id']}  |  title: {r['title']}")
        print(f"  {r['text'][:160]}...")
        print()


if __name__ == "__main__":
    main()