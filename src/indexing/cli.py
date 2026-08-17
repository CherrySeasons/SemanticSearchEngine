import argparse
from search import Retriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-ef", type=int, default=None,
                        help="Override hnsw:search_ef at query time (higher = more accurate, slower)")
    args = parser.parse_args()

    print("Loading model + index...")
    retriever = Retriever(search_ef=args.search_ef)
    print(f"Ready. top_k={args.top_k}. Type a query (or 'quit' to exit).\n")

    while True:
        query = input("query> ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        hits = retriever.search(query, top_k=args.top_k)
        for rank, hit in enumerate(hits, start=1):
            print(f"  {rank}. [{hit['score']:.3f}] {hit['title']}")
            print(f"     {hit['text'][:150]}...")
        print()


if __name__ == "__main__":
    main()
