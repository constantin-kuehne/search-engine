import argparse
import pickle
import sys
import time
from pathlib import Path

import marisa_trie


def convert_pickle_to_marisa(input_path: Path, output_path: Path) -> None:
    print(f"Loading pickle index from {input_path}...")
    start = time.time()

    with open(input_path, "rb") as f:
        index: dict[str, int] = pickle.load(f)

    load_time = time.time() - start
    print(f"Loaded {len(index):,} terms in {load_time:.2f}s")

    pickle_size = input_path.stat().st_size
    print(f"Pickle file size: {pickle_size / (1024 * 1024):.2f} MB")

    print("\nBuilding marisa-trie...")
    start = time.time()

    items = [(term, (offset,)) for term, offset in index.items()]
    trie = marisa_trie.RecordTrie("<Q", items)

    build_time = time.time() - start
    print(f"Built trie in {build_time:.2f}s")

    trie.save(str(output_path))

    marisa_size = output_path.stat().st_size
    print(f"Marisa file size: {marisa_size / (1024 * 1024):.2f} MB")
    print(f"Compression ratio: {pickle_size / marisa_size:.1f}x smaller")

    print("\nVerifying trie integrity...")
    sample_terms = list(index.keys())[:100]
    errors = 0
    for term in sample_terms:
        expected = index[term]
        results = trie.get(term)
        if not results or results[0][0] != expected:
            print(f"ERROR: term '{term}' expected {expected}, got {results}")
            errors += 1

    if errors == 0:
        print(f"Verified {len(sample_terms)} sample terms - all correct!")
    else:
        print(f"WARNING: {errors} verification errors!")
        sys.exit(1)

    print("Conversion complete!")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pickle term index to marisa-trie format"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("final_test/term_index_file"),
        help="Path to the pickle term_index_file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final_test/term_index.marisa"),
        help="Path for the output marisa trie file",
    )

    args = parser.parse_args()

    convert_pickle_to_marisa(args.input, args.output)


if __name__ == "__main__":
    main()
