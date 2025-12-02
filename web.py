from pathlib import Path
import time

import streamlit as st

import search_engine
from search_engine.inverted_index import SearchMode


@st.cache_resource
def create_index():
    final_dir =    Path("./final/")
    index = search_engine.InvertedIndex(
        final_dir / "doc_id_file_merged_final",
        final_dir / "position_list_file_merged_final",
        final_dir / "term_index_file",
        final_dir / "corpus_offset_file",
        "./msmarco-docs.tsv",
    )
    return index


num_return = 10
index = create_index()

st.title("A Search Engine")
query = st.text_input("Enter your query:")

if query:
    start = time.time()

    num_results, results = index.search(
        query, mode=SearchMode.QUERY_EVALUATOR, num_return=10, length_body=200
    )
    end = time.time()

    st.write(
        f"We found {num_results} results matching your query in {end - start:.4f} seconds."
    )
    st.write(f"{min([num_return, num_results])} of them are:")

    for result in results:
        st.divider()
        st.write(f"DocId: {result.original_docid} ({result.url}) - {result.title}\n")
        st.write(result.body)
