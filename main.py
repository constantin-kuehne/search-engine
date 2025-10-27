import search_engine
import pprint

if __name__ == "__main__":
    query = input("Enter your search query: ")

    index = search_engine.InvertedIndex()
    for row in search_engine.ingestion.process_data("./msmarco-docs.tsv", max_rows=15_000):
        index.add_document(int(row.docid[1:]), row.tokens)
    print("Indexing complete.")

    results = index.search(query)
    pprint.pprint(results)
