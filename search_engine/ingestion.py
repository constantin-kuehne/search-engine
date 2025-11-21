import csv
import sys
from typing import Generator, NamedTuple
from search_engine.preprocessing import tokenize_text

csv.field_size_limit(sys.maxsize)


class PROCESSED_ROW(NamedTuple):
    docid: int
    original_docid: str
    url: str
    title: str
    tokens: list[str]


def process_data(
    file_path: str, max_rows: int | None = None
) -> Generator[PROCESSED_ROW, None, None]:
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(
            file, delimiter="\t", fieldnames=["docid", "url", "title", "body"]
        )
        for i, row in enumerate(reader):
            if i % 1_000 == 0:
                print(f"Processed {i} rows")

            if max_rows is not None and i >= max_rows:
                break

            tokens = tokenize_text(row["body"])
            # doc_id = int(row["docid"][1:])
            yield PROCESSED_ROW(i, row["docid"], row["url"], row["title"], tokens)


if __name__ == "__main__":
    process_data("./msmarco-docs.tsv")
