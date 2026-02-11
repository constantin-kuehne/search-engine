# %%
import polars as pl
from pathlib import Path
import pickle
from search_engine.preprocessing import tokenize_text

# %%
from search_engine.inverted_index import InvertedIndex

final_dir = Path("./final/")
index = InvertedIndex(
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

# %%
from typing import Optional
from search_engine.utils import INT_SIZE, get_length_from_bytes


# %%
def calculate_term_weight(
    tf: int,
    doc_length: int,
    avg_length: float,
    b: float = 0.75,
) -> float:
    return tf / (1 - b + b * (doc_length / avg_length))


# %%
def fielded_bm25_score(
    idf_tokens: list[float],
    tf_tokens: list[float],
    k: float = 1.6,
) -> float:
    score = 0.0
    for idf_token, tf_token in zip(idf_tokens, tf_tokens):
        if idf_token == 0.0:
            continue
        score += idf_token * (tf_token * (k + 1)) / (tf_token + k)
    return score


# %%
def get_idf(
    index: InvertedIndex,
    token: str,
) -> float:
    res: Optional[int] = index.index.get(token, None)
    if res is not None:
        length_term: int = get_length_from_bytes(index.mm_doc_id_list, res)
        res += INT_SIZE + length_term  # move to the document list
        length_doc_list: int = get_length_from_bytes(index.mm_doc_id_list, res)
        idf_value: float = index.calculate_idf(
            index.metadata["num_docs"], length_doc_list
        )
        return idf_value
    else:
        return 0.0


# %%
with open(final_dir / "index_metadata", "rb") as f:
    index_metadata = pickle.load(f)

# %%
datasets = [
    "train-with-negatives.tsv",
    "train-with-negative-docs-1.tsv",
    "train-with-negative-docs-2.tsv",
    "train-with-negative-docs-3.tsv",
    "train-with-negative-docs-4.tsv",
    "train-with-negative-docs-5.tsv",
    "train-with-negative-docs-6.tsv",
    "train-with-negative-docs-95.tsv",
    "train-with-negative-docs-96.tsv",
    "train-with-negative-docs-97.tsv",
    "train-with-negative-docs-98.tsv",
    "train-with-negative-docs-99.tsv",
    "train-with-negative-docs-100.tsv",
]


import time

# %%
for dataset in datasets:
    print(f"Processing dataset {dataset}...")
    start = time.time()

    train = pl.scan_csv(f"./{dataset}", separator="\t")

    if dataset.startswith("train-with-negative-docs"):
        number = dataset.split("-")[-1].split(".")[0]
        print(f"Renaming docid {number} column... ")
        train = train.drop(["docid"])
        train = train.rename({f"{number}": "docid"})

    # %%
    train = train.with_columns(
        pl.col("body")
        .map_elements(lambda x: tokenize_text(x), return_dtype=pl.List(pl.String))
        .alias("tokenized_body")
    )

    # %%
    train = train.with_columns(
        pl.col("query")
        .map_elements(lambda x: tokenize_text(x), return_dtype=pl.List(pl.String))
        .alias("tokenized_query")
    )

    # %%
    train = train.with_columns(
        pl.col("title")
        .map_elements(lambda x: tokenize_text(x), return_dtype=pl.List(pl.String))
        .alias("tokenized_title")
    )

    # %%
    avg_body_length = index_metadata["average_doc_length"]
    avg_title_length = index_metadata["average_title_length"]
    num_docs = index_metadata["num_docs"]

    # %%
    train = train.with_columns(
        pl.col("tokenized_body")
        .map_elements(lambda x: len(x), return_dtype=pl.Int64)
        .alias("body_length"),
        pl.col("tokenized_query")
        .map_elements(lambda x: len(x), return_dtype=pl.Int64)
        .alias("query_length"),
        pl.col("tokenized_title")
        .map_elements(lambda x: len(x), return_dtype=pl.Int64)
        .alias("title_length"),
    )

    # %%
    train = train.with_columns(
        pl.col("tokenized_body").fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
    )

    # %%
    train = train.with_columns(
        pl.struct(["tokenized_query", "tokenized_body"])
        .map_elements(
            lambda s: [
                calculate_term_weight(
                    tf=s["tokenized_body"].count(token),
                    doc_length=len(s["tokenized_body"]),
                    avg_length=avg_body_length,
                )
                for token in s["tokenized_query"]
            ],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("body_term_weights"),
    )

    # %%
    train = train.with_columns(
        pl.col("tokenized_title").fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
    )

    # %%
    train = train.with_columns(
        pl.struct(["tokenized_query", "tokenized_title"])
        .map_elements(
            lambda s: [
                calculate_term_weight(
                    tf=s["tokenized_title"].count(token),
                    doc_length=len(s["tokenized_title"]),
                    avg_length=avg_title_length,
                )
                for token in s["tokenized_query"]
            ],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("title_term_weights"),
    )

    # %%
    weight_title = 2.0

    train = train.with_columns(
        term_weights=pl.struct(
            ["body_term_weights", "title_term_weights"]
        ).map_elements(
            lambda s: [
                b + t * weight_title
                for b, t in zip(s["body_term_weights"], s["title_term_weights"])
            ],
            return_dtype=pl.List(pl.Float64),
        )
    )

    # %%
    train = train.with_columns(
        pl.col("tokenized_query")
        .map_elements(
            lambda tokens: [get_idf(index, token) for token in tokens],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("query_idf_values"),
    )

    # %%
    train = train.with_columns(
        pl.struct(["term_weights", "query_idf_values"])
        .map_elements(
            lambda s: float(
                fielded_bm25_score(
                    idf_tokens=s["query_idf_values"],
                    tf_tokens=s["term_weights"],
                )
            ),
            return_dtype=pl.Float64,
        )
        .alias("bm25_score")
    )

    # %%
    train = train.with_columns(
        pl.struct(["body_term_weights", "query_idf_values"])
        .map_elements(
            lambda s: float(
                fielded_bm25_score(
                    idf_tokens=s["query_idf_values"],
                    tf_tokens=s["body_term_weights"],
                )
            ),
            return_dtype=pl.Float64,
        )
        .alias("bm25_score_body")
    )

    # %%
    train = train.with_columns(
        pl.struct(["title_term_weights", "query_idf_values"])
        .map_elements(
            lambda s: float(
                fielded_bm25_score(
                    idf_tokens=s["query_idf_values"],
                    tf_tokens=s["title_term_weights"],
                )
            ),
            return_dtype=pl.Float64,
        )
        .alias("bm25_score_title")
    )

    # %%
    train = train.with_columns(
        pl.struct(["tokenized_body", "tokenized_query", "body_length"])
        .map_elements(
            lambda s: [
                (
                    s["tokenized_body"].index(token) / s["body_length"]
                    if token in s["tokenized_body"]
                    else 1.0
                )
                for token in s["tokenized_query"]
            ],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("body_first_occurrence_indices")
    )

    # %%
    train = train.with_columns(
        pl.struct(["tokenized_title", "tokenized_query", "title_length"])
        .map_elements(
            lambda s: [
                (
                    s["tokenized_title"].index(token) / s["title_length"]
                    if token in s["tokenized_title"]
                    else 1.0
                )
                for token in s["tokenized_query"]
            ],
            return_dtype=pl.List(pl.Float64),
        )
        .alias("title_first_occurrence_indices")
    )

    # %%
    train = train.with_columns(
        body_first_occurrence_mean=pl.col("body_first_occurrence_indices").map_elements(
            lambda lst: lst.mean()
        ),
        title_first_occurrence_mean=pl.col(
            "title_first_occurrence_indices"
        ).map_elements(lambda lst: lst.mean()),
        body_first_occurrence_min=pl.col("body_first_occurrence_indices").map_elements(
            lambda lst: lst.min()
        ),
        title_first_occurrence_min=pl.col(
            "title_first_occurrence_indices"
        ).map_elements(lambda lst: lst.min()),
        in_title=pl.col("title_first_occurrence_indices").map_elements(
            lambda lst: int(any(idx < 1.0 for idx in lst))
        ),
    )

    # %%
    train = train.with_columns(
        title_length_norm=pl.col("title_length") / pl.col("title_length").max(),
        body_length_norm=pl.col("body_length") / pl.col("body_length").max(),
    )

    # %%

    # %%
    features_out = (
        [
            "index",
            "queryid",
            "docid",
            "match",
            "bm25_score",
            "bm25_score_body",
            "bm25_score_title",
            "body_first_occurrence_mean",
            "title_first_occurrence_mean",
            "body_first_occurrence_min",
            "title_first_occurrence_min",
            "body_length_norm",
            "title_length_norm",
            "in_title",
        ]
        if dataset == "train-with-negatives.tsv"
        else [
            "index",
            "queryid",
            "docid",
            "bm25_score",
            "bm25_score_body",
            "bm25_score_title",
            "body_first_occurrence_mean",
            "title_first_occurrence_mean",
            "body_first_occurrence_min",
            "title_first_occurrence_min",
            "body_length_norm",
            "title_length_norm",
            "in_title",
        ]
    )

    # %%
    train_features = train.select(features_out)

    # %%
    path = Path(dataset)
    train_features.sink_parquet(f"{path.stem}.parquet")

    end = time.time()
    print(f"Finished processing {dataset} in {end - start:.2f} seconds.")
