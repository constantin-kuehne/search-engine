import argparse
import time
from pathlib import Path

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

    final_dir = Path("./final/")

    print("Loading inverted index from disk...")
    start = time.time()
    index = search_engine.InvertedIndex(
        final_dir / "doc_id_file_merged_final",
        final_dir / "position_list_file_merged_final",
        final_dir / "term_index_file",
        final_dir / "corpus_offset_file",
        "./msmarco-docs.tsv",
    )
    end = time.time()
    print(f"Index loaded. Took {end - start:.4f}s\n")

    # print("Starting indexing...")
    # start = time.time()

    # for row in search_engine.ingestion.process_data(
    #     "./msmarco-docs.tsv", max_rows=1_000
    # ):
    #     index.add_document(
    #         row.docid, row.original_docid, row.url, row.title, row.tokens
    #     )

    # end = time.time()
    # print(f"Indexing complete. Took {end - start:.4f}s\n")

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
