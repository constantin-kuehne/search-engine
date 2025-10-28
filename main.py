import time
import search_engine
import argparse

from search_engine.inverted_index import SearchMode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Inverted Index Search Engine")
    parser.add_argument(
        "--mode",
        type=SearchMode,
        choices=[SearchMode.AND, SearchMode.OR, SearchMode.NOT, SearchMode.PHRASE],
        default=SearchMode.AND,
        help=f"Search mode: {SearchMode.AND} or {SearchMode.OR} or {SearchMode.NOT} or {SearchMode.PHRASE}",
    )
    args = parser.parse_args()
    print(args.mode)

    print("Starting indexing...")
    index = search_engine.InvertedIndex()
    for row in search_engine.ingestion.process_data("./msmarco-docs.tsv", max_rows=1_000):
        index.add_document(int(row.docid[1:]), row.url, row.title, row.tokens)

    print("Indexing complete.\n")

    query = input("Enter your search query: ")

    start = time.time()

    num_return = 10
    num_results, results = index.search(query, mode=args.mode, num_return=num_return)
    print(f"We found {num_results} results matching your query.")
    print(f"{num_return} of them are:")
    for result in results:
        print(f"DocId: {result.doc_id} ({result.url}) - {result.title}")

    end = time.time()
    print(f"\nSearch took {end - start:.4f} seconds.")
