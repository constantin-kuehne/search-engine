import csv
import heapq
import mmap
import os
import pickle
import shutil
import struct
import sys
import time
from array import array
from io import BufferedReader
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from pathlib import Path
from shutil import copyfileobj
from typing import Generator, NamedTuple

import search_engine
from search_engine.preprocessing import tokenize_text
from search_engine.utils import (INT_SIZE, LAST_UNICODE_CODE_POINT, LONG_SIZE,
                                 POSTING, get_length_from_bytes, get_trigrams_from_token)

csv.field_size_limit(sys.maxsize)


class InvertedIndexIngestion:
    def __init__(
        self,
    ) -> None:
        self.index: dict[
            str, POSTING
        ] = {}  # term -> (document list, [postion list], [position list title])
        # TODO: use list instead of dict for document ids
        self.index_title: dict[str, POSTING] = {}

        # simplemma + woosh

        self.cumulative_lengths = [0, 0]  # cumulative length for body and title
        self.max_lengths = [0, 0]  # max length for body and title

        self.term_index: dict[str, int] = {}
        self.document_lengths: array = array("I")
        if self.document_lengths.itemsize != 4:
            self.document_lengths = array("L")
            if self.document_lengths.itemsize != 4:
                raise RuntimeError("Machine does not have an exact 4 byte integer type")

        self.title_lengths: array = array("I")
        if self.title_lengths.itemsize != 4:
            self.title_lengths = array("L")
            if self.title_lengths.itemsize != 4:
                raise RuntimeError("Machine does not have an exact 4 byte integer type")

        self.doc_info_offset: array = array("I")
        if self.doc_info_offset.itemsize != 4:
            self.doc_info_offset = array("L")
            if self.doc_info_offset.itemsize != 4:
                raise RuntimeError("Machine does not have an exact 4 byte integer type")

        # trigram -> tokens and number of trigrams in token
        self.trigram_to_tokens: dict[str, set[tuple[str, int]]] = {}

    def save_to_disk(
        self,
        file_path_doc_id: str | Path,
        file_path_position_list: str | Path,
        file_path_document_lengths: str | Path,
        file_path_title_lengths: str | Path,
        file_path_doc_info_offset: str | Path,
        trigrams_path: str | Path,
        trigrams_offset_path: str | Path
    ) -> None:
        if isinstance(file_path_doc_id, str):
            file_path_doc_id = Path(file_path_doc_id)
        if isinstance(file_path_position_list, str):
            file_path_position_list = Path(file_path_position_list)
        if isinstance(file_path_document_lengths, str):
            file_path_document_lengths = Path(file_path_document_lengths)
        if isinstance(file_path_title_lengths, str):
            file_path_title_lengths = Path(file_path_title_lengths)
        if isinstance(file_path_doc_info_offset, str):
            file_path_doc_info_offset = Path(file_path_doc_info_offset)

        file_path_doc_id.parent.mkdir(parents=True, exist_ok=True)
        file_path_position_list.parent.mkdir(parents=True, exist_ok=True)
        file_path_document_lengths.parent.mkdir(parents=True, exist_ok=True)
        file_path_title_lengths.parent.mkdir(parents=True, exist_ok=True)

        doc_id_file = open(file_path_doc_id, mode="wb")
        position_list_file = open(file_path_position_list, mode="wb")
        document_length_file = open(file_path_document_lengths, mode="wb")
        title_length_file = open(file_path_title_lengths, mode="wb")
        doc_info_offset_file = open(file_path_doc_info_offset, mode="wb")

        document_length_file.write(self.document_lengths)
        document_length_file.close()

        title_length_file.write(self.title_lengths)
        title_length_file.close()

        doc_info_offset_file.write(self.doc_info_offset)
        doc_info_offset_file.close()

        self.write_trigrams(trigrams_path, trigrams_offset_path)

        sorted_index_keys = sorted(self.index.keys())
        for term in sorted_index_keys:
            doc_list, position_list_list, position_list_list_title = self.index[term]

            document_index = []

            term_bytes = term.encode("utf-8")
            # term length
            doc_id_file.write(struct.pack("I", len(term_bytes)))
            # term bytes
            doc_id_file.write(term_bytes)
            # document frequency
            doc_id_file.write(struct.pack("I", len(doc_list)))
            # document list
            doc_id_file.write(struct.pack(f"{len(doc_list)}I", *doc_list))

            term_frequencies: list[int] = []
            term_frequencies_title: list[int] = []

            assert len(position_list_list) == len(position_list_list_title)

            for position_list, position_list_title in zip(
                position_list_list, position_list_list_title
            ):
                document_index.append(position_list_file.tell())

                position_list_combined = position_list_title + position_list

                position_list_file.write(struct.pack("I", len(position_list_combined)))
                term_frequencies.append(len(position_list))
                term_frequencies_title.append(len(position_list_title))

                position_list_file.write(
                    struct.pack(
                        f"{len(position_list_combined)}I", *position_list_combined
                    )
                )

            assert len(document_index) == len(doc_list)

            # term frequencies title
            doc_id_file.write(
                struct.pack(f"{len(term_frequencies_title)}I", *term_frequencies_title)
            )

            # term frequencies
            doc_id_file.write(
                struct.pack(f"{len(term_frequencies)}I", *term_frequencies)
            )
            # position list offsets
            doc_id_file.write(struct.pack(f"{len(document_index)}Q", *document_index))

            # doc id file: term length | term bytes | doc freq | [doc list] | [term frequencies] | [position list offsets]
        # doc id title: term length | term bytes | doc freq | [doc list] | [term frequencies title] | [term frequencies] | [position list offsets]

        # modify position list to include title positions = [title pos 1, title pos 2, body pos 1, body pos2]
        # read by using the term frequencies title to separate them

        del self.index
        self.index = {}

        del self.trigram_to_tokens
        self.trigram_to_tokens = {}

        doc_id_file.close()
        position_list_file.close()

    def save_term_index(self, file_path_term_index: str | Path):
        with open(file_path_term_index, "wb") as f:
            pickle.dump(self.term_index, f)

    @staticmethod
    def __get_position_list_bytes_and_offsets(
        current_offset: int,
        bytes_position_list_index,
        length_doc_list: int,
        index: int,
        mm_position_files: list[mmap.mmap],
    ) -> tuple[bytes, list[int]]:
        # TODO: remove term frequencies and calulate outside function by combining the lists
        offsets_pos_list: list[int] = []

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
        )

    def process_trigrams_in_row(
        self,
        tokens: list[str]
    ) -> None:
        for token in tokens:
            trigrams = get_trigrams_from_token(token)

            for trigram in trigrams:
                if trigram not in self.trigram_to_tokens:
                    self.trigram_to_tokens[trigram] = set()

                self.trigram_to_tokens[trigram].add((token, len(trigrams)))

    def write_trigrams(
        self,
        trigrams_path: Path,
        trigram_offsets_path: Path
    ) -> None:
        trigrams = open(trigrams_path, "wb")
        trigram_offsets = open(trigram_offsets_path, "wb")

        current_offset: int = 0

        sorted_trigrams = sorted(self.trigram_to_tokens.keys())
        for trigram in sorted_trigrams:
            encoded_trigram = trigram.encode("utf-8")
            trigram_offsets.write(struct.pack("I", len(encoded_trigram)))
            trigram_offsets.write(encoded_trigram)
            trigrams.write(struct.pack("I", len(self.trigram_to_tokens[trigram])))
            current_offset += 4
            for token, num_trigrams_in_token in self.trigram_to_tokens[trigram]:
                encoded_token = token.encode("utf-8")
                trigrams.write(struct.pack("I", len(encoded_token)))
                trigrams.write(encoded_token)
                trigrams.write(struct.pack("I", num_trigrams_in_token))
                current_offset += 4 + len(encoded_token) + 4

    def merge_trigrams(
        self,
        trigrams_path: Path,
        trigram_offsets_path: Path,
        src_path_stem: Path,
        src_trigrams_prefix: str,
        src_trigrams_offsets_prefix: str,
        num_blocks: int
    ) -> None:
        trigrams: list = []
        trigram_offsets: list = []

        out_trigrams = open(trigrams_path, "wb")
        out_trigrams_offsets = open(trigram_offsets_path, "wb")

        for block_id in range(num_blocks):
            trigrams.append(open(src_path_stem / (src_trigrams_prefix + str(block_id)), "rb"))
            trigram_offsets.append(open(src_path_stem / (src_trigrams_offsets_prefix + str(block_id)), "rb"))

        min_heap = []

        for block_id, trigram_offset in enumerate(trigram_offsets):
            trigram_length = struct.unpack("I", trigram_offset.read(4))[0]
            trigram: str = trigram_offset.read(trigram_length).decode("utf-8")
            heapq.heappush(min_heap, (trigram, block_id))

        current_min: str = ""
        current_min_blocks: list[int] = []
        tokens: set[bytes] = set()
        current_out_offset: int = 0
        first_element: bool = True
        while min_heap:
            trigram, block_id = heapq.heappop(min_heap)
            if first_element or trigram == current_min:
                current_min_blocks.append(block_id)
                if first_element:
                    first_element = False
                    current_min = trigram
            else:
                encoded_trigram = current_min.encode("utf-8")
                out_trigrams_offsets.write(struct.pack("I", len(encoded_trigram)))
                out_trigrams_offsets.write(encoded_trigram)
                out_trigrams_offsets.write(struct.pack("Q", current_out_offset))

                for i, this_block_id in enumerate(current_min_blocks):
                    num_tokens = struct.unpack("I", trigrams[this_block_id].read(4))[0]
                    for token_id in range(num_tokens):
                        token_length_bytes = trigrams[this_block_id].read(4)
                        token_length = struct.unpack("I", token_length_bytes)[0]
                        token_bytes: bytes = trigrams[this_block_id].read(token_length + 4) # +4 for number of trigrams in token

                        tokens.add(token_length_bytes + token_bytes)

                sum_tokens: int = len(tokens)
                out_trigrams.write(struct.pack("I", sum_tokens))
                current_out_offset += 4

                for token_with_length_bytes_and_num_trigrams_in_token in tokens:
                    out_trigrams.write(token_with_length_bytes_and_num_trigrams_in_token)
                    current_out_offset += len(token_with_length_bytes_and_num_trigrams_in_token)

                current_min = trigram
                current_min_blocks.clear()
                current_min_blocks.append(block_id)
                tokens.clear()

            offset_bytes: bytes = trigram_offsets[block_id].read(4)
            if len(offset_bytes) == 0:
                continue
            next_trigram_length: int = struct.unpack("I", offset_bytes)[0]
            new_trigram: str = trigram_offsets[block_id].read(next_trigram_length).decode("utf-8")
            heapq.heappush(min_heap, (new_trigram, block_id))

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

        doc_id_file = open(file_path_doc_id_merged, "wb")
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
                + length_doc_list * INT_SIZE * 3 : offset_doc_list
                + INT_SIZE
                + length_doc_list
                * INT_SIZE
                * 3  # times 3 because we want to skip the doc id list and the term frequencies title list and term frequencies list
                + length_doc_list * LONG_SIZE
            ]

            bytes_position_list, offsets_pos_list = (
                self.__get_position_list_bytes_and_offsets(
                    position_list_file.tell(),
                    bytes_position_list_index,
                    length_doc_list,
                    index,
                    mm_position_files,
                )
            )

            term_frequencies_title: bytes = mm_files[index][
                offset_doc_list
                + INT_SIZE
                + length_doc_list * INT_SIZE : offset_doc_list
                + INT_SIZE
                + length_doc_list * INT_SIZE
                + length_doc_list * INT_SIZE
            ]

            term_frequencies: bytes = mm_files[index][
                offset_doc_list
                + INT_SIZE
                + length_doc_list * INT_SIZE
                + length_doc_list * INT_SIZE : offset_doc_list
                + INT_SIZE
                + length_doc_list * INT_SIZE
                + length_doc_list * INT_SIZE
                + length_doc_list * INT_SIZE
            ]

            file_positions[index] = (
                offset_doc_list
                + INT_SIZE
                + length_doc_list
                * INT_SIZE
                * 3  # times 3 becaue of the doc list, term frequencies title list and term frequencies list
                + length_doc_list * LONG_SIZE  # this one is the postion list index
            )

            if file_positions[index] >= mm_files[index].size():
                file_ended[index] = True

            for index in min_indices[1:]:
                pos = file_positions[index]

                offset_doc_list_local = pos + INT_SIZE + length_term
                length_doc_list_local: int = get_length_from_bytes(
                    mm_files[index], offset_doc_list_local
                )
                length_doc_list += length_doc_list_local

            doc_id_file.write(bytes_in_file[0 : INT_SIZE + length_term])
            doc_id_file.write(struct.pack("I", length_doc_list))

            doc_id_file.write(bytes_in_file[INT_SIZE + INT_SIZE + length_term :])
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
                    + length_doc_list_local * INT_SIZE * 3 : offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local * INT_SIZE * 3
                    + length_doc_list_local * LONG_SIZE
                ]

                (
                    bytes_position_list_local,
                    offsets_pos_list_local,
                ) = self.__get_position_list_bytes_and_offsets(
                    position_list_file.tell(),
                    bytes_position_list_index_local,
                    length_doc_list_local,
                    index,
                    mm_position_files,
                )

                offsets_pos_list += offsets_pos_list_local
                position_list_file.write(bytes_position_list_local)

                term_frequencies_title_local: bytes = mm_files[index][
                    offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local * INT_SIZE : offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local * INT_SIZE
                    + length_doc_list_local * INT_SIZE
                ]

                term_frequencies_local: bytes = mm_files[index][
                    offset_doc_list
                    + INT_SIZE
                    + length_doc_list * INT_SIZE
                    + length_doc_list * INT_SIZE : offset_doc_list
                    + INT_SIZE
                    + length_doc_list * INT_SIZE
                    + length_doc_list * INT_SIZE
                    + length_doc_list * INT_SIZE
                ]

                term_frequencies_title += term_frequencies_title_local
                term_frequencies += term_frequencies_local

                file_positions[index] = (
                    offset_doc_list_local
                    + INT_SIZE
                    + length_doc_list_local
                    * INT_SIZE
                    * 3  # times 3 becaue of the doc list, term frequencies title list and term frequencies list
                    + length_doc_list_local
                    * LONG_SIZE  # this one is the postion list index
                )

                if file_positions[index] >= mm_files[index].size():
                    file_ended[index] = True

                doc_id_file.write(bytes_in_file)

            assert (
                len(struct.unpack(f"{length_doc_list}I", term_frequencies))
                == length_doc_list
            )
            assert len(offsets_pos_list) == length_doc_list

            doc_id_file.write(term_frequencies_title)
            doc_id_file.write(term_frequencies)

            doc_id_file.write(
                struct.pack(f"{len(offsets_pos_list)}Q", *offsets_pos_list)
            )

        for file_handle in file_handles:
            file_handle.close()

        doc_id_file.close()

        for file_handle in file_handles_position:
            file_handle.close()

        position_list_file.close()

    def add_document(
        self, doc_id: int, tokens: list[str], tokens_title: list[str]
    ) -> None:
        doc_length = len(tokens)
        self.document_lengths.append(doc_length)
        self.cumulative_lengths[0] += doc_length
        self.max_lengths[0] = max(self.max_lengths[0], doc_length)

        title_length = len(tokens_title)
        self.title_lengths.append(title_length)
        self.cumulative_lengths[1] += title_length
        self.max_lengths[1] = max(self.max_lengths[1], title_length)

        length = doc_length + title_length

        for position in range(length):
            if position < title_length:
                term = tokens_title[position]
                if term not in self.index:
                    self.index[term] = ([doc_id], [[]], [[position]])
                else:
                    doc_list, position_list_list, position_list_list_title = self.index[
                        term
                    ]
                    if doc_list[-1] != doc_id:
                        doc_list.append(doc_id)
                        position_list_list.append([])
                        position_list_list_title.append([position])
                    else:
                        position_list_list_title[-1].append(position)
            else:
                position -= len(tokens_title)
                term = tokens[position]
                if term not in self.index:
                    self.index[term] = ([doc_id], [[position]], [[]])
                else:
                    doc_list, position_list_list, position_list_list_title = self.index[
                        term
                    ]
                    if doc_list[-1] != doc_id:
                        doc_list.append(doc_id)
                        position_list_list.append([position])
                        position_list_list_title.append([])
                    else:
                        position_list_list[-1].append(position)

    def merge_contiguous_files(
        self,
        dest_path: str | Path,
        src_path_stem: Path,
        src_filename_prefix: str,
        num_blocks: int,
    ) -> None:
        with open(dest_path, mode="wb") as dest:
            for source_path in [
                src_path_stem / (src_filename_prefix + str(block_id))
                for block_id in range(num_blocks)
            ]:
                with open(source_path, mode="rb") as source:
                    copyfileobj(source, dest)

    def merge_offsets(
        self,
        dest_path: str | Path,
        src_path_stem: Path,
        src_filename_prefix: str,
        num_blocks: int,
        INT_SIZE: int,
    ) -> None:
        if INT_SIZE != 4 and INT_SIZE != 8:
            raise RuntimeError("Not implemented yet!")

        if INT_SIZE == 4:
            wanted_array_format_char = "I"
            fallback_array_format_char = "L"
        if INT_SIZE == 8:
            wanted_array_format_char = "L"
            fallback_array_format_char = "Q"

        with open(dest_path, mode="wb") as dest:
            with open(src_path_stem / (src_filename_prefix + "0"), mode="rb") as source:
                copyfileobj(source, dest)
                source.seek(-INT_SIZE, os.SEEK_END)
                last_chunk_highest_offset: int = int.from_bytes(
                    source.peek(INT_SIZE), signed=False, byteorder="little"
                )  # this is the offset at which the next chunk starts

            for source_path in [
                src_path_stem / (src_filename_prefix + str(block_id))
                for block_id in range(1, num_blocks)
            ]:
                with open(source_path, mode="rb") as source:
                    file_size: int = os.path.getsize(source_path)
                    num_items: int = file_size // INT_SIZE
                    block_array = array(wanted_array_format_char)
                    block_array.fromfile(source, num_items)
                    for i in range(0, num_items):
                        block_array[i] = block_array[i] + last_chunk_highest_offset

                    last_chunk_highest_offset = block_array[-1]
                    dest.seek(
                        -INT_SIZE, os.SEEK_CUR
                    )  # overwrite the last index written in the last block (the first index of this block), as array.tofile writes the full array
                    block_array.tofile(dest)

    def merge_doc_info_offsets(
        self,
        dest_path: str | Path,
        src_path_stem: Path,
        src_filename_prefix: str,
        num_blocks: int,
    ) -> None:
        self.merge_offsets(dest_path, src_path_stem, src_filename_prefix, num_blocks, 4)

    def merge_bodies_offsets(
        self,
        dest_path: str | Path,
        src_path_stem: Path,
        src_filename_prefix: str,
        num_blocks: int,
    ) -> None:
        self.merge_offsets(dest_path, src_path_stem, src_filename_prefix, num_blocks, 8)


class PROCESSED_ROW(NamedTuple):
    docid: int
    # tokens: list[str]
    line: str


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

            yield pos, PROCESSED_ROW(i, line)

            pos = file.tell()
            line = file.readline()
            i += 1


def process_chunk(
    chunk: list[PROCESSED_ROW], block_num: int, blocks_dir: Path
) -> tuple[list[int], list[int]]:
    local_index = InvertedIndexIngestion()

    doc_info_file = open(blocks_dir / f"doc_info_file_{block_num}", "wb")
    bodies_file = open(blocks_dir / f"bodies_{block_num}", "wb")
    bodies_offsets_file = open(blocks_dir / f"bodies_offsets_{block_num}", "wb")
    bodies_file_pos = 0

    for doc in chunk:
        row: list[str] = next(csv.reader([doc.line], delimiter="\t"))
        tokens: list[str] = tokenize_text(row[3])
        tokens_title: list[str] = tokenize_text(row[2])

        local_index.process_trigrams_in_row(tokens)

        local_index.add_document(doc.docid, tokens, tokens_title)
        local_index.doc_info_offset.append(doc_info_file.tell())
        doc_info_file.write("\t".join([row[0], row[1], row[2]]).encode("utf-8"))
        encoded_body = row[3].encode("utf-8")
        bodies_file.write(encoded_body)
        bodies_offsets_file.write(struct.pack("Q", bodies_file_pos))
        bodies_file_pos += len(encoded_body)

    # append the current index once more, i.e., the index directly after all values from this chunk (like C++ end() iterator)
    # this is the index at which the next chunk starts and is used during the merge
    local_index.doc_info_offset.append(doc_info_file.tell())
    bodies_offsets_file.write(struct.pack("Q", bodies_file_pos))
    doc_info_file.close()

    local_index.save_to_disk(
        blocks_dir / f"doc_id_files/doc_id_file_block_{block_num}",
        blocks_dir / f"position_list_files/position_list_file_block_{block_num}",
        blocks_dir / f"document_lengths_{block_num}",
        blocks_dir / f"title_lengths_{block_num}",
        blocks_dir / f"doc_info_offsets_{block_num}",
        blocks_dir / f"trigrams_{block_num}",
        blocks_dir / f"trigram_offsets_{block_num}"
    )
    return local_index.cumulative_lengths, local_index.max_lengths


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
    final_dir = Path("./final_test/")

    shutil.rmtree(staged_dir, ignore_errors=True)
    shutil.rmtree(blocks_dir, ignore_errors=True)

    blocks_dir.mkdir(parents=True, exist_ok=True)
    (blocks_dir / "doc_id_files/").mkdir(parents=True, exist_ok=True)
    (blocks_dir / "position_list_files/").mkdir(parents=True, exist_ok=True)

    staged_dir.mkdir(parents=True, exist_ok=True)
    (staged_dir / "doc_id_files/").mkdir(parents=True, exist_ok=True)
    (staged_dir / "position_list_files/").mkdir(parents=True, exist_ok=True)

    shutil.rmtree(final_dir, ignore_errors=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    index = InvertedIndexIngestion()

    print("Starting indexing...")
    start = time.time()
    block_num = 0

    block_size = 15_000
    max_rows = None
    # max_rows = 25

    num_processes: int = (os.cpu_count() or 6) - 2

    print("Starting processing rows...")
    chunk = []

    num_docs = 0
    cumulative_length = 0
    cumulative_length_title = 0
    max_doc_length = 0
    max_title_length = 0

    threads: list[AsyncResult] = []

    with Pool(processes=num_processes) as pool:
        print(f"Starting processing rows with a pool of {num_processes} processes...")
        for pos, row in search_engine.ingestion.process_data(
            "./msmarco-docs.tsv", max_rows=max_rows
        ):
            chunk.append(row)

            num_docs += 1
            if row.docid > 0 and row.docid % block_size == 0:
                if len(threads) == num_processes:
                    has_ready_thread: bool = False
                    current_index: int = 0
                    while not has_ready_thread:
                        if threads[current_index].ready():
                            has_ready_thread = True
                            break
                        current_index += 1
                        if current_index == num_processes:
                            current_index = 0
                            time.sleep(1)

                    cumulative_lengths_block, max_lengths  = threads[current_index].get()
                    cumulative_length += cumulative_lengths_block[0]
                    cumulative_length += cumulative_lengths_block[1]
                    max_doc_length = max(max_doc_length, max_lengths[0])
                    max_title_length = max(max_title_length, max_lengths[1])


                    threads.pop(current_index)

                    threads.append(pool.apply_async(process_chunk, args=(chunk, block_num, blocks_dir), error_callback=lambda x: print(x)))
                else:
                    threads.append(pool.apply_async(process_chunk, args=(chunk, block_num, blocks_dir), error_callback=lambda x: print(x)))

                chunk = []
                block_num += 1

        if chunk:
            threads.append(
                pool.apply_async(
                    process_chunk,
                    args=(chunk, block_num, blocks_dir),
                    error_callback=lambda e: print(f"Error: {e}"),
                )
            )

        del chunk

        pool.close()
        pool.join()

    for thread in threads:
        cumulative_lengths_block, max_lengths = thread.get()
        cumulative_length += cumulative_lengths_block[0]
        cumulative_length_title += cumulative_lengths_block[1]
        max_doc_length = max(max_doc_length, max_lengths[0])
        max_title_length = max(max_title_length, max_lengths[1])

    average_doc_length = cumulative_length / num_docs
    average_title_length = cumulative_length_title / num_docs
    meta_data = {
        "max_doc_length": max_doc_length,
        "max_title_length": max_title_length,
        "average_doc_length": average_doc_length,
        "average_title_length": average_title_length,
        "num_docs": num_docs,
    }

    with open(final_dir / "index_metadata", "wb") as f:
        pickle.dump(meta_data, f)

    print(f"Finished processing rows in {time.time() - start:.4f}s")

    print("Starting merging blocks...")
    start_merge = time.time()

    num_files = len(os.listdir(blocks_dir / "doc_id_files/"))
    length_one_stage = min(num_processes, num_files)

    index.merge_contiguous_files(
        final_dir / "document_lengths", blocks_dir, "document_lengths_", num_files
    )
    index.merge_contiguous_files(
        final_dir / "title_lengths", blocks_dir, "title_lengths_", num_files
    )
    index.merge_contiguous_files(
        final_dir / "doc_info_file", blocks_dir, "doc_info_file_", num_files
    )
    index.merge_contiguous_files(final_dir / "bodies", blocks_dir, "bodies_", num_files)

    index.merge_doc_info_offsets(
        final_dir / "doc_info_offsets", blocks_dir, "doc_info_offsets_", num_files
    )
    index.merge_bodies_offsets(
        final_dir / "bodies_offsets", blocks_dir, "bodies_offsets_", num_files
    )
    index.merge_trigrams(
        final_dir / "trigrams", final_dir / "trigram_offsets", blocks_dir, "trigrams_", "trigram_offsets_", num_files
    )

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
            # merge_blocks(current_start, current_end, i, staged_dir, blocks_dir)
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

    print(f"Finished merging blocks in {time.time() - start_merge:.4f}s")

    shutil.rmtree(staged_dir)
    shutil.rmtree(blocks_dir)

    index.save_term_index(final_dir / "term_index_file")

    end = time.time()
    print(f"Indexing complete. Took {end - start:.4f}s\n")
