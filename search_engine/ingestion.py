import csv
import mmap
import os
import pickle
import struct
import sys
import time
from io import BufferedReader
from pathlib import Path
from typing import Generator, NamedTuple

import search_engine
from search_engine.preprocessing import tokenize_text
from search_engine.utils import (INT_SIZE, LAST_UNICODE_CODE_POINT,
                                 LAST_UTF8_CODE_POINT, POSTING, DocumentInfo,
                                 SearchMode, get_length_from_bytes)

csv.field_size_limit(sys.maxsize)


class InvertedIndexIngestion:
    def __init__(
        self,
    ) -> None:
        self.index: dict[str, POSTING] = {}  # term -> (document list, [postion list])
        # TODO: use list instead of dict for document ids

        # simplemma + woosh

        self.corpus_offset: dict[int, int] = {}
        self.term_index: dict[str, int] = {}

    def save_to_disk(
        self,
        file_path_doc_id: str,
        file_path_position_list: str,
        file_path_position_index: str,
    ) -> None:
        Path(file_path_doc_id).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path_position_list).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path_position_index).parent.mkdir(parents=True, exist_ok=True)

        doc_id_file = open(file_path_doc_id, "w+b")
        position_list_file = open(file_path_position_list, mode="wb")
        position_index_file = open(file_path_position_index, mode="wb")

        for term, (doc_list, position_list_list) in sorted(self.index.items()):
            document_index = []

            term_bytes = term.encode("utf-8")
            doc_id_file.write(struct.pack("I", len(term_bytes)))
            doc_id_file.write(term_bytes)
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

        del self.index
        self.index = {}

        doc_id_file.close()
        position_list_file.close()
        position_index_file.close()

    def save_corpus_offset(self, file_path_corpus_offset: str):
        with open(file_path_corpus_offset, "wb") as f:
            pickle.dump(self.corpus_offset, f)

    def save_term_index(self, file_path_term_index: str):
        with open(file_path_term_index, "wb") as f:
            pickle.dump(self.term_index, f)

    def merge_blocks(
        self,
        file_path_doc_id_merged: str,
        file_path_block_dir: str,
        doc_id_file_pattern: str,
    ):
        num_files = len(os.listdir(file_path_block_dir))

        doc_id_file = open(file_path_doc_id_merged, "w+b")

        file_handlers: list[BufferedReader] = []

        mm_files: list[mmap.mmap] = []
        file_positions: list[int] = []
        file_ended: list[bool] = [False] * num_files

        for i in range(num_files):
            file_path = Path(file_path_block_dir) / f"{doc_id_file_pattern}{i}"
            file_handler = file_path.open("rb")
            file_handlers.append(file_handler)
            mm_files.append(
                mmap.mmap(file_handler.fileno(), length=0, prot=mmap.PROT_READ)
            )
            file_positions.append(file_handler.tell())

        term_list: list[str] = [LAST_UNICODE_CODE_POINT] * num_files
        while not all(file_ended):
            current_min = LAST_UNICODE_CODE_POINT
            for i, mm_file in enumerate(mm_files):
                if file_ended[i]:
                    continue

                length_term: int = get_length_from_bytes(mm_file, file_positions[i])
                term = mm_file[
                    file_positions[i] + INT_SIZE : file_positions[i]
                    + INT_SIZE
                    + length_term
                ].decode(encoding="utf-8")
                term_list[i] = term
                if min(current_min, term) == term:
                    current_min = term

            min_indices = [i for i, term in enumerate(term_list) if term == current_min]

            length_term: int = len(current_min.encode("utf-8"))
            pos = file_positions[min_indices[0]]

            offset_doc_list = pos + INT_SIZE + length_term
            length_doc_list: int = get_length_from_bytes(
                mm_files[min_indices[0]], offset_doc_list
            )

            bytes_in_file = mm_files[min_indices[0]][
                pos : offset_doc_list + INT_SIZE + length_doc_list * INT_SIZE
            ]
            doc_id_file_pos = doc_id_file.tell()
            self.term_index[current_min] = doc_id_file_pos

            file_positions[min_indices[0]] = (
                offset_doc_list + INT_SIZE + length_doc_list * INT_SIZE
            )

            if file_positions[min_indices[0]] >= mm_files[min_indices[0]].size():
                file_ended[min_indices[0]] = True

            doc_id_file.write(bytes_in_file)

            for index in min_indices[1:]:
                pos = file_positions[min_indices[index]]

                offset_doc_list_local = pos + INT_SIZE + length_term
                length_doc_list_local: int = get_length_from_bytes(
                    mm_files[index], offset_doc_list_local
                )

                bytes_in_file = mm_files[
                    index
                ][
                    offset_doc_list_local + INT_SIZE : offset_doc_list_local
                    + INT_SIZE  # we only want to write the doc list therefore we skip the int representing its length
                    + length_doc_list_local * INT_SIZE
                ]
                length_doc_list += length_doc_list_local
                file_positions[index] = (
                    offset_doc_list_local + INT_SIZE + length_doc_list_local * INT_SIZE
                )

                if file_positions[index] >= mm_files[index].size():
                    file_ended[index] = True

                doc_id_file.write(bytes_in_file)

            doc_id_file_end_pos = doc_id_file.tell()

            doc_id_file.seek(doc_id_file_pos + INT_SIZE + length_term)
            doc_id_file.write(struct.pack("I", length_doc_list))
            doc_id_file.seek(doc_id_file_end_pos)

        for file_handler in file_handlers:
            file_handler.close()

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
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
    tokens: list[str]


def process_data(
    file_path: str, max_rows: int | None = None
) -> Generator[tuple[int, PROCESSED_ROW], None, None]:
    with open(file_path, mode="r", encoding="utf-8") as file:
        i = 0
        pos = file.tell()
        line = file.readline()
        while line:
            if i % 1_000 == 0:
                print(f"Processed {i} rows")

            if max_rows is not None and i >= max_rows:
                break

            row = next(csv.reader([line], delimiter="\t"))
            tokens = tokenize_text(row[3])

            yield pos, PROCESSED_ROW(i, tokens)

            pos = file.tell()
            line = file.readline()
            i += 1


if __name__ == "__main__":
    index = InvertedIndexIngestion()

    print("Starting indexing...")
    start = time.time()

    block_size = 100
    block_num = 0

    for pos, row in search_engine.ingestion.process_data(
        "./msmarco-docs.tsv", max_rows=201
    ):
        if row.docid > 0 and row.docid % block_size == 0:
            index.save_to_disk(
                f"./blocks/doc_id_files/doc_id_file_block_{block_num}",
                f"./blocks/position_list_files/position_list_file_block_{block_num}",
                f"./blocks/position_list_indices/position_list_index_block_{block_num}",
            )
            block_num += 1

        index.add_document(row.docid, row.tokens)

        index.corpus_offset[row.docid] = pos

    index.merge_blocks(
        "./doc_id_file_merged", "./blocks/doc_id_files/", "doc_id_file_block_"
    )

    index.save_corpus_offset("./corpus_offset_file")
    index.save_term_index("./term_index_file")

    end = time.time()
    print(f"Indexing complete. Took {end - start:.4f}s\n")
