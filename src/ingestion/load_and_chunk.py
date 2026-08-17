"""
Phase 1 - Ingestion
Load Simple English Wikipedia, clean text, chunk into ~200-word passages
with overlap, and write to data/processed/chunks.jsonl.

Usage:
    python src/ingestion/load_and_chunk.py [--max-articles N]
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_TARGET_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 30


def clean_text(text: str) -> str:
    """Strip residual citation markers / excess whitespace left in the
    HF plaintext extraction."""
    text = re.sub(r"\[\d+\]", "", text)          # [1], [23] citation markers
    text = re.sub(r"\n{2,}", "\n\n", text)        # collapse blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)        # collapse repeated spaces
    return text.strip()


def split_into_paragraphs(text: str):
    return [p.strip() for p in text.split("\n") if p.strip()]


def chunk_paragraphs(paragraphs, target_words=CHUNK_TARGET_WORDS,
                    overlap_words=CHUNK_OVERLAP_WORDS):
    """Greedily pack paragraphs into ~target_words chunks, sliding back
    overlap_words on each new chunk for context continuity."""
    chunks = []
    buffer = []
    for para in paragraphs:
        buffer.extend(para.split())
        while len(buffer) >= target_words:
            chunk_words = buffer[:target_words]
            chunks.append(" ".join(chunk_words))
            buffer = buffer[target_words - overlap_words:]
    if len(buffer) >= MIN_CHUNK_WORDS:
        chunks.append(" ".join(buffer))
    return chunks


def process_article(article_id, title, text):
    text = clean_text(text)
    paragraphs = split_into_paragraphs(text)
    raw_chunks = chunk_paragraphs(paragraphs)
    return [
        {
            "chunk_id": f"{article_id}_{i}",
            "article_id": article_id,
            "title": title,
            "chunk_index": i,
            "text": chunk,
            "num_words": len(chunk.split()),
        }
        for i, chunk in enumerate(raw_chunks)
    ]


def main(max_articles=None):
    print("Loading Simple English Wikipedia dataset (wikimedia/wikipedia, 20231101.simple)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    if max_articles:
        ds = ds.select(range(max_articles))

    all_chunks = []
    for idx, row in enumerate(ds):
        all_chunks.extend(process_article(row["id"], row["title"], row["text"]))
        if idx % 1000 == 0:
            print(f"  processed {idx} articles -> {len(all_chunks)} chunks so far")

    out_path = OUTPUT_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_chunks)} chunks written to {out_path}")
    print(f"Avg chunk length: {sum(r['num_words'] for r in all_chunks) / len(all_chunks):.1f} words")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit number of articles (useful for a quick test run)")
    args = parser.parse_args()
    main(max_articles=args.max_articles)
