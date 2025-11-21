import argparse
import time

import search_engine
from search_engine.inverted_index import SearchMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Inverted Index Search Engine")
    parser.add_argument(
        "--mode",
        type=SearchMode,
        choices=[x for x in SearchMode],
        default=SearchMode.AND,
        required=True,
        help=f"Search mode: {' or '.join([str(x) for x in SearchMode])}",
    )
    parser.add_argument(
        "--num_return",
        type=int,
        default=10,
        help="How many search results should be returned",
    )
    args = parser.parse_args()

    index = search_engine.InvertedIndex()

    print("Starting indexing...")
    start = time.time()

    for row in search_engine.ingestion.process_data(
        "./msmarco-docs.tsv", max_rows=15_000
    ):
        index.add_document(row.docid, row.original_docid, row.url, row.title, row.tokens)

    end = time.time()
    print(f"Indexing complete. Took {end - start:.4f}s\n")

    try:
        while True:
            query = input("Enter your search query: ")

            start = time.time()

            num_results, results = index.search(
                query, mode=args.mode, num_return=args.num_return
            )
            print(f"We found {num_results} results matching your query.")
            print(f"{args.num_return} of them are:")
            for result in results:
                print(f"DocId: {result.original_docid} ({result.url}) - {result.title}")

            end = time.time()
            print(f"\nSearch took {end - start:.4f} seconds.")
    except KeyboardInterrupt:
        print("\nprogram terminated")
