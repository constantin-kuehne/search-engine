import csv
import mmap
import os
import pickle
import struct
import sys
import time
from io import BufferedReader
from multiprocessing import Pool
from pathlib import Path
from typing import Generator, NamedTuple

import search_engine
from search_engine.preprocessing import tokenize_text
from search_engine.utils import (INT_SIZE, LAST_UNICODE_CODE_POINT,
                                 LAST_UTF8_CODE_POINT, LONG_SIZE, POSTING,
                                 DocumentInfo, SearchMode,
                                 get_length_from_bytes)

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
        file_path_doc_id: str | Path,
        file_path_position_list: str | Path,
    ) -> None:
        if isinstance(file_path_doc_id, str):
            file_path_doc_id = Path(file_path_doc_id)
        if isinstance(file_path_position_list, str):
            file_path_position_list = Path(file_path_position_list)

        file_path_doc_id.parent.mkdir(parents=True, exist_ok=True)
        file_path_position_list.parent.mkdir(parents=True, exist_ok=True)

        doc_id_file = open(file_path_doc_id, "w+b")
        position_list_file = open(file_path_position_list, mode="wb")

        for term, (doc_list, position_list_list) in sorted(self.index.items()):
            document_index = []

            term_bytes = term.encode("utf-8")
            doc_id_file.write(struct.pack("I", len(term_bytes)))
            doc_id_file.write(term_bytes)
            doc_id_file.write(struct.pack("I", len(doc_list)))
            doc_id_file.write(struct.pack(f"{len(doc_list)}I", *doc_list))

            term_frequencies: list[int] = []
            for position_list in position_list_list:
                document_index.append(position_list_file.tell())

                position_list_file.write(struct.pack("I", len(position_list)))
                term_frequencies.append(len(position_list))

                position_list_file.write(
                    struct.pack(f"{len(position_list)}I", *position_list)
                )

            assert len(document_index) == len(doc_list)
            # {
            #     "echo": offset_into_doc_id_file
            #     "test": offset_into_doc_id_file2
            # }

            # |
            # v
            # len(term) term len(doc_id_list)[doc_id, doc_id, doc_id][offset_pos_list, offset_pos_list, offset_pos_list]

            # |
            # v
            # len(term2) term2 len(docid_list2)[doc_id2, doc_id2, doc_id2][offset_pos_list2, offset_pos_list2, offset_pos_list2][tf, tf, tf]
            # offset -> index [pos_list1_offset, pos_list2_offset, pos_list3_offset] -> x*len(pos_List)

            # |
            # v
            # len(pos_list) [1, 4, 9]

            doc_id_file.write(
                struct.pack(f"{len(term_frequencies)}I", *term_frequencies)
            )
            doc_id_file.write(struct.pack(f"{len(document_index)}Q", *document_index))

        del self.index
        self.index = {}

        doc_id_file.close()
        position_list_file.close()

    def save_corpus_offset(self, file_path_corpus_offset: str | Path):
        with open(file_path_corpus_offset, "wb") as f:
            pickle.dump(self.corpus_offset, f)

    def save_term_index(self, file_path_term_index: str | Path):
        with open(file_path_term_index, "wb") as f:
            pickle.dump(self.term_index, f)

    @staticmethod
    def __get_position_list_bytes_and_offsets_and_term_frequencies(
        current_offset: int,
        bytes_position_list_index,
        length_doc_list: int,
        index: int,
        mm_position_files: list[mmap.mmap],
    ) -> tuple[bytes, list[int], list[int]]:
        offsets_pos_list: list[int] = []
        term_frequencies: list[int] = []

        first_position_list_offset = struct.unpack(
            "Q",
            bytes_position_list_index[0:LONG_SIZE],
        )[0]

        offsets_pos_list.append(current_offset)

        current_block_pos_offset = first_position_list_offset
        accumulated_byte_size = 0

        for i in range(length_doc_list):
            length_pos_list_local = struct.unpack(
                "I",
                mm_position_files[index][
                    current_block_pos_offset : current_block_pos_offset + INT_SIZE
                ],
            )[0]
            term_frequencies.append(length_pos_list_local)

            size_of_current_pos_list_block = INT_SIZE + length_pos_list_local * INT_SIZE

            current_block_pos_offset += size_of_current_pos_list_block

            accumulated_byte_size += size_of_current_pos_list_block

            if i < length_doc_list - 1:
                offsets_pos_list.append(current_offset + accumulated_byte_size)

        assert len(offsets_pos_list) == length_doc_list

        return (
            mm_position_files[index][
                first_position_list_offset:current_block_pos_offset
            ],
            offsets_pos_list,
            term_frequencies,
        )

    def merge_blocks(
        self,
        file_path_doc_id_merged: str | Path,
        file_path_doc_id_block_dir: str | Path,
        doc_id_file_pattern: str,
        file_path_position_list_merged: str | Path,
        file_path_position_list_block_dir: str | Path,
        position_list_file_pattern: str,
        start_block: int,
        end_block: int,
    ):
        Path(file_path_doc_id_merged).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path_position_list_merged).parent.mkdir(parents=True, exist_ok=True)

        doc_id_file = open(file_path_doc_id_merged, "w+b")
        position_list_file = open(file_path_position_list_merged, "wb")

        file_handles: list[BufferedReader] = []
        file_handles_position: list[BufferedReader] = []

        mm_files: list[mmap.mmap] = []
        file_positions: list[int] = []
        num_files = end_block - start_block
        file_ended: list[bool] = [False] * num_files

        mm_position_files: list[mmap.mmap] = []
        position_file_positions: list[int] = []

        for i in range(start_block, end_block):
            file_path = Path(file_path_doc_id_block_dir) / f"{doc_id_file_pattern}{i}"
            file_handle = file_path.open("rb")
            file_handles.append(file_handle)
            mm_files.append(
                mmap.mmap(file_handle.fileno(), length=0, prot=mmap.PROT_READ)
            )
            file_positions.append(file_handle.tell())

            position_file_path = (
                Path(file_path_position_list_block_dir)
                / f"{position_list_file_pattern}{i}"
            )
            file_handle_position = position_file_path.open("rb")
            file_handles_position.append(file_handle_position)
            mm_position_files.append(
                mmap.mmap(file_handle_position.fileno(), length=0, prot=mmap.PROT_READ)
            )
            position_file_positions.append(file_handle_position.tell())

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
            index = min_indices[0]

            length_term: int = len(current_min.encode("utf-8"))
            pos = file_positions[index]

            offset_doc_list = pos + INT_SIZE + length_term
            length_doc_list: int = get_length_from_bytes(
                mm_files[index], offset_doc_list
            )

            bytes_in_file = mm_files[index][
                pos : offset_doc_list + INT_SIZE + length_doc_list * INT_SIZE
            ]
            doc_id_file_pos = doc_id_file.tell()
            self.term_index[current_min] = doc_id_file_pos

            bytes_position_list_index = mm_files[
                index
            ][
                offset_doc_list
                + INT_SIZE
                + length_doc_list * INT_SIZE * 2 : offset_doc_list
                + INT_SIZE
                + length_doc_list
                * INT_SIZE
                * 2  # times 2 because we want to skip the doc id list and the term frequencies list
                + length_doc_list * LONG_SIZE
            ]

            bytes_position_list, offsets_pos_list, term_frequencies = (
                self.__get_position_list_bytes_and_offsets_and_term_frequencies(
                    position_list_file.tell(),
                    bytes_position_list_index,
                    length_doc_list,
                    index,
                    mm_position_files,
                )
            )

            file_positions[index] = (
                offset_doc_list
                + INT_SIZE
                + length_doc_list
                * INT_SIZE
                * 2  # times two becaue of the doc list and term frequencies list
                + length_doc_list * LONG_SIZE  # this one is the postion list index
            )

            if file_positions[index] >= mm_files[index].size():
                file_ended[index] = True

            doc_id_file.write(bytes_in_file)
            position_list_file.write(bytes_position_list)

            for index in min_indices[1:]:
                pos = file_positions[index]

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

                bytes_position_list_index_local = mm_files[index][
                    offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local * INT_SIZE * 2 : offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local * INT_SIZE * 2
                    + length_doc_list_local * LONG_SIZE
                ]

                (
                    bytes_position_list_local,
                    offsets_pos_list_local,
                    term_frequencies_local,
                ) = self.__get_position_list_bytes_and_offsets_and_term_frequencies(
                    position_list_file.tell(),
                    bytes_position_list_index_local,
                    length_doc_list_local,
                    index,
                    mm_position_files,
                )
                offsets_pos_list += offsets_pos_list_local
                position_list_file.write(bytes_position_list_local)
                term_frequencies += term_frequencies_local

                length_doc_list += length_doc_list_local
                file_positions[index] = (
                    offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local
                    * INT_SIZE
                    * 2  # times two becaue of the doc list and term frequencies list
                    + length_doc_list_local
                    * LONG_SIZE  # this one is the postion list index
                )

                if file_positions[index] >= mm_files[index].size():
                    file_ended[index] = True

                doc_id_file.write(bytes_in_file)

            assert len(term_frequencies) == length_doc_list
            assert len(offsets_pos_list) == length_doc_list

            doc_id_file.write(
                struct.pack(f"{len(term_frequencies)}I", *term_frequencies)
            )

            doc_id_file.write(
                struct.pack(f"{len(offsets_pos_list)}Q", *offsets_pos_list)
            )

            doc_id_file_end_pos = doc_id_file.tell()

            # go back and write the merged length of the doc list
            doc_id_file.seek(doc_id_file_pos + INT_SIZE + length_term)
            doc_id_file.write(struct.pack("I", length_doc_list))
            doc_id_file.seek(doc_id_file_end_pos)

        for file_handle in file_handles:
            file_handle.close()

        doc_id_file.close()

        for file_handle in file_handles_position:
            file_handle.close()

        position_list_file.close()

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
            if i % 10_000 == 0:
                print(f"Processed {i} rows")

            if max_rows is not None and i >= max_rows:
                break

            row = next(csv.reader([line], delimiter="\t"))
            tokens = tokenize_text(row[3])

            yield pos, PROCESSED_ROW(i, tokens)

            pos = file.tell()
            line = file.readline()
            i += 1


def process_chunk(chunk: list[PROCESSED_ROW], block_num: int, blocks_dir: Path) -> None:
    local_index = InvertedIndexIngestion()

    for row in chunk:
        local_index.add_document(row.docid, row.tokens)

    local_index.save_to_disk(
        blocks_dir / f"doc_id_files/doc_id_file_block_{block_num}",
        blocks_dir / f"position_list_files/position_list_file_block_{block_num}",
    )
    del local_index


def merge_blocks(
    current_start: int, current_end: int, i: int, staged_dir: Path, blocks_dir: Path
) -> None:
    local_index = InvertedIndexIngestion()
    local_index.merge_blocks(
        staged_dir / f"doc_id_files/doc_id_file_merged_{i}",
        blocks_dir / "doc_id_files/",
        "doc_id_file_block_",
        staged_dir / f"position_list_files/position_list_file_merged_{i}",
        blocks_dir / "position_list_files/",
        "position_list_file_block_",
        current_start,
        current_end,
    )
    del local_index


if __name__ == "__main__":
    blocks_dir = Path("./blocks/")
    staged_dir = Path("./staged/")
    final_dir = Path("./final/")

    blocks_dir.mkdir(parents=True, exist_ok=True)
    (blocks_dir / "doc_id_files/").mkdir(parents=True, exist_ok=True)
    (blocks_dir / "position_list_files/").mkdir(parents=True, exist_ok=True)
    (staged_dir / "doc_id_files/").mkdir(parents=True, exist_ok=True)
    (staged_dir / "position_list_files/").mkdir(parents=True, exist_ok=True)

    final_dir.mkdir(parents=True, exist_ok=True)
    staged_dir.mkdir(parents=True, exist_ok=True)



    # blocks_dir = Path("./blocks_little/")
    # staged_dir = Path("./staged_little/")
    # final_dir = Path("./final_little/")

    index = InvertedIndexIngestion()

    print("Starting indexing...")
    start = time.time()

    block_size = 7_500
    # block_size = 500
    block_num = 0

    max_rows = 3_400_000
    # max_rows = 15_001

    num_processes = (os.cpu_count() or 6) - 2

    print("Starting processing rows...")
    chunk = []

    document_lengths: dict[int, int] = {}
    num_docs = 0
    cumulative_length = 0

    with Pool(processes=num_processes) as pool:
        print(f"Starting processing rows with a pool of {num_processes} processes...")
        for pos, row in search_engine.ingestion.process_data(
            "./msmarco-docs.tsv", max_rows=max_rows
        ):
            chunk.append(row)
            document_lengths[row.docid] = len(row.tokens)
            cumulative_length += len(row.tokens)
            num_docs += 1
            if row.docid > 0 and row.docid % block_size == 0:
                pool.apply_async(process_chunk, args=(chunk, block_num, blocks_dir))

                chunk = []
                block_num += 1

            index.corpus_offset[row.docid] = pos

        if chunk:
            pool.apply_async(process_chunk, args=(chunk, block_num, blocks_dir))

        del chunk

        pool.close()
        pool.join()

    with open(final_dir / "document_lengths", "wb") as f:
        pickle.dump(document_lengths, f)

    average_doc_length = cumulative_length / num_docs
    meta_data = {"average_doc_length": average_doc_length, "num_docs": num_docs}

    with open(final_dir / "index_metadata", "wb") as f:
        pickle.dump(meta_data, f)

    print(f"Finished processing rows in {time.time() - start:.4f}s")

    print("Starting merging blocks...")
    start_merge = time.time()

    num_files = len(os.listdir(blocks_dir / "doc_id_files/"))
    length_one_stage = min(num_processes, num_files)

    current_start = 0
    with Pool(processes=num_processes) as pool:
        print(f"Starting merging files with a pool of {num_processes} processes...")
        for i, current_end in enumerate(
            range(length_one_stage, num_files + 1, length_one_stage)
        ):
            pool.apply_async(
                merge_blocks,
                args=(current_start, current_end, i, staged_dir, blocks_dir),
            )
            current_start = current_end
        pool.close()
        pool.join()

    i = num_files // length_one_stage
    rest_files = num_files % length_one_stage
    if rest_files > 0 and rest_files < length_one_stage:
        index.merge_blocks(
            staged_dir / f"doc_id_files/doc_id_file_merged_{i}",
            blocks_dir / "doc_id_files/",
            "doc_id_file_block_",
            staged_dir / f"position_list_files/position_list_file_merged_{i}",
            blocks_dir / "position_list_files/",
            "position_list_file_block_",
            current_start,
            num_files,
        )

    num_files = len(os.listdir(staged_dir / "doc_id_files/"))
    if num_files >= 2:
        index.merge_blocks(
            final_dir / "doc_id_file_merged_final",
            staged_dir / "doc_id_files/",
            "doc_id_file_merged_",
            final_dir / "position_list_file_merged_final",
            staged_dir / "position_list_files/",
            "position_list_file_merged_",
            0,
            num_files,
        )
    else:
        os.rename(
            staged_dir / "doc_id_files/doc_id_file_merged_0",
            final_dir / "doc_id_file_merged_final",
        )
        os.rename(
            staged_dir / "position_list_files/position_list_file_merged_0",
            final_dir / "position_list_file_merged_final",
        )

    print(f"Finished merging blocks in {time.time() - start_merge:.4f}s")

    index.save_corpus_offset(final_dir / "corpus_offset_file")
    index.save_term_index(final_dir / "term_index_file")

    end = time.time()
    print(f"Indexing complete. Took {end - start:.4f}s\n")
