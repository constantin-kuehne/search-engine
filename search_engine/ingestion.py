import csv
import pickle
import struct
import sys
import time
from typing import Generator, NamedTuple

import search_engine
from search_engine.preprocessing import tokenize_text
from search_engine.utils import POSTING, DocumentInfo, SearchMode

csv.field_size_limit(sys.maxsize)


class InvertedIndexIngestion:
    def __init__(
        self,
    ) -> None:
        self.index: dict[str, POSTING] = {}  # term -> (document list, [postion list])
        # TODO: use list instead of dict for document ids

        # simplemma + woosh

        self.docs: dict[int, DocumentInfo] = {}

    def save_to_disk(
        self,
        file_path_doc_id: str,
        file_path_position_list: str,
        file_path_position_index: str,
        file_path_term_index: str,
    ) -> None:
        doc_id_file = open(file_path_doc_id, "w+b")
        term_index_file = open(file_path_term_index, "wb")
        position_list_file = open(file_path_position_list, mode="wb")
        position_index_file = open(file_path_position_index, mode="wb")

        term_index: dict[str, tuple[int, int]] = {}

        for term, (doc_list, position_list_list) in sorted(self.index.items()):
            term_index[term] = (doc_id_file.tell(), position_index_file.tell())
            document_index = []

            doc_id_file.write(struct.pack("I", len(term)))
            doc_id_file.write(term.encode("utf-8"))
            doc_id_file.write(struct.pack("I", len(doc_list)))
            doc_id_file.write(struct.pack(f"{len(doc_list)}I", *doc_list))

            for position_list in position_list_list:
                document_index.append(position_list_file.tell())

                position_list_file.write(struct.pack("I", len(position_list)))
                position_list_file.write(
                    struct.pack(f"{len(position_list)}I", *position_list)
                )

            position_index_file.write(
                struct.pack(f"{len(document_index)}I", *document_index)
            )

        pickle.dump(term_index, term_index_file)

        del self.index
        self.index = {}

        doc_id_file.close()
        term_index_file.close()
        position_list_file.close()
        position_index_file.close()

    def add_document(
        self, doc_id: int, original_docid: str, url: str, title: str, tokens: list[str]
    ) -> None:
        self.docs[doc_id] = DocumentInfo(
            original_docid=original_docid, url=url, title=title
        )
        for position, term in enumerate(tokens):
            if term not in self.index:
                self.index[term] = ([doc_id], [[position]])
            else:
                doc_list, position_list_list = self.index[term]
                if doc_list[-1] != doc_id:
                    doc_list.append(doc_id)
                    position_list_list.append([position])
                else:
                    position_list_list[-1].append(position)


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
    index = InvertedIndexIngestion()

    print("Starting indexing...")
    start = time.time()

    block_size = 7_500
    block_num = 0

    for row in search_engine.ingestion.process_data(
        "./msmarco-docs.tsv", max_rows=15_000
    ):
        if row.docid % block_size == 0:
            index.save_to_disk(
                f"./blocks/doc_id_file_block_{block_num}",
                f"./blocks/position_list_file_block_{block_num}",
                f"./blocks/position_list_index_block_{block_num}",
                f"./blocks/term_index_file_block_{block_num}",
            )
            block_num += 1

        index.add_document(
            row.docid, row.original_docid, row.url, row.title, row.tokens
        )

    end = time.time()
    print(f"Indexing complete. Took {end - start:.4f}s\n")
