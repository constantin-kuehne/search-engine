#!/usr/bin/env python
# coding: utf-8

# In[1]:
import polars as pl

# In[ ]:
docs = pl.scan_csv(
    "./msmarco-docs.tsv",
    separator="\t",
    has_header=False,
    new_columns=["docid", "url", "title", "body"],
)
queries = pl.scan_csv(
    "./msmarco-doctrain-queries.tsv",
    separator="\t",
    has_header=False,
    new_columns=["queryid", "query"],
)

# In[ ]:
# Join documents with relevance dataset
query_rel = pl.scan_csv(
    "./msmarco-doctrain-qrels.tsv",
    separator=" ",
    has_header=False,
    new_columns=["queryid", "dummy", "docid", "rel"],
)
train_joined = docs.join(query_rel, on="docid").join(queries, on="queryid")

# In[1]:
train_joined.sink_csv("train.tsv", separator="\t")


# In[2]:
train = pl.read_csv(
    "./train.tsv",
    separator="\t",
    has_header=True,
)

# In[4]:
top100 = pl.read_csv(
    "./msmarco-doctrain-top100_old.tsv",
    separator=" ",
    has_header=False,
    new_columns=["queryid", "dummy", "docid", "rank", "score", "algorithm"],
)


# In[5]:
top_bot10 = top100.filter((pl.col("rank") <= 6) | (pl.col("rank") > 94))


# In[6]:
# make columns for each rank
top_bot10_pivoted = top_bot10.pivot(on="rank", index="queryid", values="docid")


# In[7]:
train_negatives = train.join(top_bot10_pivoted, on="queryid")

# In[8]:
target_cols = ["1", "2", "3", "4", "5", "6", "95", "96", "97", "98", "99", "100"]

# In[ ]:
train_negatives = train_negatives.with_columns(
    match=pl.coalesce(
        [pl.when(pl.col("docid") == pl.col(c)).then(pl.lit(c)) for c in target_cols]
    )
)

# In[ ]:
train_negatives = train_negatives.with_row_index()

# In[11]:
train_negatives.write_csv(
    "./train-with-negatives.tsv",
    separator="\t",
    include_header=True,
)

# In[ ]:
train_negatives_lazy = pl.scan_csv(
    "./train-with-negatives.tsv",
    separator="\t",
    has_header=True,
)

# In[ ]:
train_negatives_lazy = train_negatives_lazy.select(
    target_cols + ["index", "docid", "queryid", "query"]
)

# In[ ]:
train_negatives_lazy_copy = train_negatives_lazy.clone()

# In[ ]:
for c in target_cols:
    # train_negatives_lazy_copy = docs.join(
    #     train_negatives_lazy, left_on="docid", right_on=c, suffix=f"_{c}"
    # )
    print(f"Processing column {c}...")

    train_negatives_lazy_copy = train_negatives_lazy.join(
        docs, left_on=c, right_on="docid", suffix=f"_{c}"
    )

    train_negatives_lazy_copy.sink_csv(
        f"./train-with-negative-docs-{c}.tsv",
        separator="\t",
        include_header=True,
    )
# %%
