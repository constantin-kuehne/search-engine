import argparse
import readline  # noqa: F401 to enable arrow key history navigation
import time
from pathlib import Path
from shutil import get_terminal_size

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
    parser.add_argument(
        "--length_body",
        type=int,
        default=100,
        help="Number of characters to show in the body snippet of each search result",
    )

    args = parser.parse_args()

    final_dir = Path("./final/")

    print("Loading inverted index from disk...")
    start = time.time()
    index = search_engine.InvertedIndex(
        final_dir / "doc_id_file_merged_final",
        final_dir / "position_list_file_merged_final",
        final_dir / "term_index_file",
        final_dir / "doc_info_offsets",
        final_dir / "doc_info_file",
        final_dir / "index_metadata",
        final_dir / "document_lengths",
        final_dir / "title_lengths",
        final_dir / "bodies",
        final_dir / "bodies_offsets",
    )
    end = time.time()
    print(f"Index loaded. Took {end - start:.4f}s\n")

    try:
        while True:
            query = input("Enter your search query: ")

            start = time.time()

            num_results, results = index.search(
                query,
                mode=args.mode,
                num_return=args.num_return,
                snippet_length=args.length_body
            )
            print(f"We found {num_results} results matching your query.")
            print(f"{min(args.num_return, num_results)} of them are:")

            num_terminal_columns = get_terminal_size().columns
            for score, search_result in results:
                print("-" * num_terminal_columns)
                print(
                    f"Score: {score} - DocId: {search_result.original_docid} ({search_result.url}) - {search_result.title}"
                )
                print(f"Body ({args.length_body} chars): {search_result.body_snippet}")

            end = time.time()
            print(f"\nSearch took {end - start:.4f} seconds.")
    except KeyboardInterrupt:
        print("\nProgram terminated.")
