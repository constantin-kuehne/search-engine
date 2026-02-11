#!/usr/bin/env python
# coding: utf-8

# %%
import polars as pl

# %%
docs = pl.scan_csv(
    "./msmarco-docs.tsv",
    separator="\t",
    has_header=False,
    new_columns=["docid", "url", "title", "body"],
)
queries = pl.scan_csv(
    "./msmarco-docdev-queries.tsv",
    separator="\t",
    has_header=False,
    new_columns=["queryid", "query"],
)

# %%
# Join documents with relevance dataset
query_rel = pl.scan_csv(
    "./msmarco-docdev-qrels.tsv",
    separator=" ",
    has_header=False,
    new_columns=["queryid", "dummy", "docid", "rel"],
)
train_joined = docs.join(query_rel, on="docid").join(queries, on="queryid")

# %%
train_joined.sink_csv("dev.tsv", separator="\t")


# %%
train = pl.read_csv(
    "./dev.tsv",
    separator="\t",
    has_header=True,
)

# %%
top100 = pl.read_csv(
    "./msmarco-docdev-top100",
    separator=" ",
    has_header=False,
    new_columns=["queryid", "dummy", "docid", "rank", "score", "algorithm"],
)


# %%
# make columns for each rank
top100_pivoted = top100.pivot(on="rank", index="queryid", values="docid")


# %%
train_negatives = train.join(top100_pivoted, on="queryid")

# %%
target_cols = [f"{i}" for i in range(1, 101)]

# %%
train_negatives = train_negatives.with_columns(
    match=pl.coalesce(
        [pl.when(pl.col("docid") == pl.col(c)).then(pl.lit(c)) for c in target_cols]
    )
)

# %%
train_negatives = train_negatives.with_row_index()

# In[11]:
train_negatives.write_csv(
    "./dev-with-negatives.tsv",
    separator="\t",
    include_header=True,
)

# %%
train_negatives_lazy = pl.scan_csv(
    "./dev-with-negatives.tsv",
    separator="\t",
    has_header=True,
)

# %%
train_negatives_lazy = train_negatives_lazy.select(
    target_cols + ["index", "docid", "queryid", "query"]
)

# %%
train_negatives_lazy_copy = train_negatives_lazy.clone()

# %%
for c in target_cols:
    # train_negatives_lazy_copy = docs.join(
    #     train_negatives_lazy, left_on="docid", right_on=c, suffix=f"_{c}"
    # )
    print(f"Processing column {c}...")

    train_negatives_lazy_copy = train_negatives_lazy.join(
        docs, left_on=c, right_on="docid", suffix=f"_{c}"
    )

    train_negatives_lazy_copy.sink_csv(
        f"./dev-with-negative-docs-{c}.tsv",
        separator="\t",
        include_header=True,
    )
# %%
