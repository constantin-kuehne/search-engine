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
        help=f"Search mode: {' or '.join([repr(x) for x in SearchMode])}",
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
    parser.add_argument(
        "--disable_spelling_correction",
        action="store_false",
        help="Whether disable enable spelling correction",
        dest="enable_spelling_correction",
        default=True,
    )
    parser.add_argument(
        "--disable_approximate_nearest_neighbors",
        action="store_false",
        help="Whether disable enable approximate nearest neighbors for semantic search",
        dest="enable_approximate_nearest_neighbors",
        default=True,
    )

    args = parser.parse_args()

    enable_semantic_search = args.mode == SearchMode.SEMANTIC
    print(f"Semantic search enabled: {enable_semantic_search}")

    final_dir = Path("./final_test/")

    print("Loading inverted index from disk...")
    start = time.time()
    index = search_engine.InvertedIndex(
        file_path_doc_id=final_dir / "doc_id_file_merged_final",
        file_path_position_list=final_dir / "position_list_file_merged_final",
        file_path_term_index=final_dir / "term_index.marisa",
        file_path_doc_info_offset=final_dir / "doc_info_offsets",
        file_path_doc_info=final_dir / "doc_info_file",
        file_path_metadata=final_dir / "index_metadata",
        file_path_document_lengths=final_dir / "document_lengths",
        file_path_title_lengths=final_dir / "title_lengths",
        file_path_bodies=final_dir / "bodies",
        file_path_bodies_offsets=final_dir / "bodies_offsets",
        file_path_trigrams=final_dir / "trigrams",
        file_path_trigram_offsets=final_dir / "trigram_offsets",
        file_path_ranking_model="./search_engine/ranking_model/checkpoints/1pdz89si/best_checkpoint.pth",
        file_path_embeddings="./final_embed/embeddings.npy",
        file_path_embedding_metadata="./final_embed/embedding_metadata",
        file_path_kmeans_model="./final_embed/kmeans.pkl",
        enable_semantic_search=enable_semantic_search,
        enable_spelling_correction=args.enable_spelling_correction,
        enable_approximate_nearest_neighbors=args.enable_approximate_nearest_neighbors,
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
                snippet_length=args.length_body,
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
