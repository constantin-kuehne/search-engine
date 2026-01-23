import csv
import heapq
import math
import mmap
import os
import pickle
import struct
from array import array
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import editdistance
from ordered_set import OrderedSet

from search_engine.preprocessing import (build_query_tree, shunting_yard,
                                         tokenize_text)
from search_engine.utils import (INT_SIZE, LONG_SIZE, DocumentInfo, SearchMode,
                                 get_length_from_bytes, get_trigrams_from_token)


class InvertedIndex:
    def __init__(
        self,
        file_path_doc_id: str | Path,
        file_path_position_list: str | Path,
        file_path_term_index: str | Path,
        file_path_doc_info_offset: str | Path,
        file_path_doc_info: str | Path,
        file_path_metadata: str | Path,
        file_path_document_lengths: str | Path,
        file_path_title_lengths: str | Path,
        file_path_bodies: str | Path,
        file_path_bodies_offsets: str | Path,
        file_path_trigrams: str | Path,
        file_path_trigram_offsets: str | Path
    ) -> None:
        doc_id_file = open(file_path_doc_id, "rb")
        term_index_file = open(file_path_term_index, "rb")
        position_list_file = open(file_path_position_list, mode="rb")
        bodies_file = open(file_path_bodies, mode="rb")
        bodies_offsets_file = open(file_path_bodies_offsets, mode="rb")
        trigrams_file = open(file_path_trigrams, mode="rb")

        doc_info_file = open(file_path_doc_info, "rb")

        self.index = pickle.load(term_index_file)  # TODO: Set to self.index

        self.mm_doc_id_list = mmap.mmap(
            doc_id_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        self.mm_position_list = mmap.mmap(
            position_list_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        self.mm_doc_info = mmap.mmap(
            doc_info_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        self.mm_bodies = mmap.mmap(bodies_file.fileno(), length=0, prot=mmap.PROT_READ)
        self.mm_bodies_offsets = mmap.mmap(bodies_offsets_file.fileno(), length=0, prot=mmap.PROT_READ)

        with open(file_path_doc_info_offset, "rb") as f:
            self.docs: array[int] = array('I')
            if self.docs.itemsize != 4:
                self.docs = array('L')
                if self.docs.itemsize != 4:
                    raise RuntimeError("Machine does not have an exact 4 byte integer type")
            file_bytes: int = os.path.getsize(file_path_doc_info_offset)
            self.docs.fromfile(f, file_bytes // 4)

        term_index_file.close()

        self.reader = lambda x: csv.DictReader(
            x, delimiter="\t", fieldnames=["docid", "url", "title"]
        )

        self.trigram_to_tokens: dict[str, int] = {}
        with open(file_path_trigram_offsets, "rb") as trigrams:
            while True:
                trigram_length_bytes: bytes = trigrams.read(4)
                if len(trigram_length_bytes) != 4:
                    break
                trigram_length: int = struct.unpack("I", trigram_length_bytes)[0]
                trigram: str = trigrams.read(trigram_length).decode("utf-8")
                offset: int = struct.unpack("Q", trigrams.read(8))[0]
                self.trigram_to_tokens[trigram] = offset

        self.mm_trigrams = mmap.mmap(trigrams_file.fileno(), length=0, prot=mmap.PROT_READ)

        with open(file_path_metadata, "rb") as f:
            self.metadata = pickle.load(f)

        with open(file_path_document_lengths, "rb") as f:
            file_bytes: int = os.path.getsize(file_path_document_lengths)
            self.document_lengths: array = array('I')
            if self.document_lengths.itemsize != 4:
                self.document_lengths = array('L')
                if self.document_lengths.itemsize != 4:
                    raise RuntimeError("Machine does not have an exact 4 byte integer type")
            self.document_lengths.fromfile(f, file_bytes // 4) # uint32_t

        with open(file_path_title_lengths, "rb") as f:
            file_bytes: int = os.path.getsize(file_path_document_lengths)
            self.title_lengths: array = array('I')
            if self.title_lengths.itemsize != 4:
                self.title_lengths = array('L')
                if self.title_lengths.itemsize != 4:
                    raise RuntimeError("Machine does not have an exact 4 byte integer type")
            self.title_lengths.fromfile(f, file_bytes // 4) # uint32_t

    def has_phrase(self, pos_list_list: list[tuple[int]]) -> bool:
        indices = [0 for _ in range(len(pos_list_list))]
        has_phrase: bool = False

        for _ in range(len(pos_list_list[0])):
            for i, pos_list in enumerate(pos_list_list[1:]):
                while (
                    pos_list[indices[i + 1]] <= pos_list_list[i][indices[i]]
                ):  # +1 because we skip first list
                    indices[i + 1] += 1

                    if indices[i + 1] >= len(pos_list):
                        return False

                if pos_list[indices[i + 1]] == pos_list_list[i][indices[i]] + 1:
                    has_phrase = True
                else:
                    has_phrase = False
                    break

            if has_phrase:
                break

            indices[0] += 1

        return has_phrase

    def intersection(
        self,
        doc_ids: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]]]:
        doc_ids = [doc_list for doc_list in doc_ids if len(doc_list) > 0]
        pointer = [0 for _ in range(len(doc_ids))]
        result_doc_ids = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids):
            heapq.heappush(
                min_heap,
                (
                    doc_list[0],
                    i,
                    term_frequencies[i][0],
                    term_frequencies_title[i][0],
                ),
            )

        counter_same_value = 0
        last_min = -1

        result_term_freqs: list[list[int]] = []
        last_term_freqs: list[int] = []

        result_term_freqs_title: list[list[int]] = []
        last_term_freqs_title: list[int] = []

        one_list_finished = False

        while min_heap:
            current_min, current_index, term_frequency, term_frequency_title = (
                heapq.heappop(min_heap)
            )

            if last_min == current_min:
                counter_same_value += 1
            else:
                if one_list_finished:
                    break
                last_term_freqs = []
                last_term_freqs_title = []
                counter_same_value = 0

            last_term_freqs.append(term_frequency)
            last_term_freqs_title.append(term_frequency_title)

            if counter_same_value == len(doc_ids) - 1:
                result_term_freqs.append(last_term_freqs)
                result_term_freqs_title.append(last_term_freqs_title)
                result_doc_ids.append(current_min)

            pointer[current_index] += 1
            if len(doc_ids[current_index]) <= pointer[current_index]:
                last_min = current_min
                one_list_finished = True
                continue

            heapq.heappush(
                min_heap,
                (
                    doc_ids[current_index][pointer[current_index]],
                    current_index,
                    term_frequencies[current_index][pointer[current_index]],
                    term_frequencies_title[current_index][pointer[current_index]],
                ),
            )
            last_min = current_min
        return result_doc_ids, result_term_freqs, result_term_freqs_title

    def intersection_phrase(
        self,
        doc_ids_per_token: list[tuple[int, ...]],
        doc_pos_per_token: list[tuple[int, ...]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]], Sequence[Sequence[int]]]:
        pointer = [0 for _ in range(len(doc_ids_per_token))]
        result_doc_ids: list[int] = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids_per_token):
            heapq.heappush(
                min_heap,
                (
                    doc_list[0],
                    i,
                    term_frequencies[i][0],
                    term_frequencies_title[i][0],
                    doc_pos_per_token[i][0],
                ),
            )

        counter_same_value = 0
        last_min = -1

        result_term_freqs: list[list[int]] = []
        last_term_freqs: list[int] = []

        result_term_freqs_title: list[list[int]] = []
        last_term_freqs_title: list[int] = []

        result_doc_pos_offsets: list[list[int]] = []
        last_doc_pos_offsets: list[int] = []

        one_list_finished = False

        while min_heap:
            (
                current_min,
                current_index,
                term_frequency,
                term_frequency_title,
                doc_pos_offset,
            ) = heapq.heappop(min_heap)

            if last_min == current_min:
                counter_same_value += 1
            else:
                if one_list_finished:
                    break
                last_term_freqs = []
                last_term_freqs_title = []
                last_doc_pos_offsets = []
                counter_same_value = 0

            last_term_freqs.append(term_frequency)
            last_term_freqs_title.append(term_frequency_title)
            last_doc_pos_offsets.append(doc_pos_offset)

            if counter_same_value == len(doc_ids_per_token) - 1:
                result_term_freqs.append(last_term_freqs)
                result_term_freqs_title.append(last_term_freqs_title)
                result_doc_pos_offsets.append(last_doc_pos_offsets)
                result_doc_ids.append(current_min)

            pointer[current_index] += 1
            if len(doc_ids_per_token[current_index]) <= pointer[current_index]:
                last_min = current_min
                one_list_finished = True
                continue

            heapq.heappush(
                min_heap,
                (
                    doc_ids_per_token[current_index][pointer[current_index]],
                    current_index,
                    term_frequencies[current_index][pointer[current_index]],
                    term_frequencies_title[current_index][pointer[current_index]],
                    doc_pos_per_token[current_index][pointer[current_index]],
                ),
            )
            last_min = current_min

        return (
            result_doc_ids,
            result_term_freqs,
            result_term_freqs_title,
            result_doc_pos_offsets,
        )

    def union(
        self,
        doc_ids: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]]]:
        num_terms = len(doc_ids)
        pointers = [0] * num_terms
        result_doc_ids = []
        result_term_freqs = []
        result_term_freqs_title = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids):
            if doc_list:
                heapq.heappush(min_heap, (doc_list[0], i))

        last_doc_id = -1
        current_tf_vector = []
        current_tf_vecotr_title = []
        while min_heap:
            doc_id, term_index = heapq.heappop(min_heap)
            if doc_id != last_doc_id:
                if last_doc_id != -1:
                    result_term_freqs.append(current_tf_vector)

                last_doc_id = doc_id
                result_doc_ids.append(doc_id)
                current_tf_vector = [0] * num_terms
                current_tf_vecotr_title = [0] * num_terms

            tf = term_frequencies[term_index][pointers[term_index]]
            tf_title = term_frequencies_title[term_index][pointers[term_index]]
            current_tf_vector[term_index] = tf
            current_tf_vecotr_title[term_index] = tf_title
            pointers[term_index] += 1

            if pointers[term_index] < len(doc_ids[term_index]):
                next_doc_id = doc_ids[term_index][pointers[term_index]]
                heapq.heappush(min_heap, (next_doc_id, term_index))

        if last_doc_id != -1:
            result_term_freqs.append(current_tf_vector)
            result_term_freqs_title.append(current_tf_vecotr_title)
        return result_doc_ids, result_term_freqs, result_term_freqs_title

    def and_statement(
        self,
        doc_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[Sequence[int], Sequence[Sequence[int]], Sequence[Sequence[int]]]:
        matched: tuple[
            Sequence[int], Sequence[Sequence[int]], Sequence[Sequence[int]]
        ] = tuple([])
        if len(doc_list) == 1:
            matched = (
                doc_list[0],
                list(zip(*term_frequencies)),
                list(zip(*term_frequencies_title)),
            )
        elif len(doc_list) > 1:
            matched = self.intersection(
                doc_list, term_frequencies, term_frequencies_title
            )

        return matched

    def or_statement(
        self,
        doc_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]]]:
        matched: tuple[list[int], list[list[int]], list[list[int]]] = tuple([])
        if len(doc_list) == 1:
            matched = (
                list(doc_list[0]),
                [list(term_frequencies[0])],
                [list(term_frequencies_title[0])],
            )
        elif len(doc_list) > 1:
            matched = self.union(doc_list, term_frequencies, term_frequencies_title)

        return matched

    def not_statement(
        self, doc_list: Sequence[Sequence[int]], *args
    ) -> tuple[list[int], list[list[int]], list[list[int]]]:
        matched: tuple[list[int], list[list[int]], list[list[int]]] = tuple([])
        if len(doc_list) == 0:
            term_freqs = [[0 for _ in range(len(self.docs))]]
            matched = (list(self.docs)), term_freqs, term_freqs.copy()
        else:
            doc_ids_matched = list(OrderedSet(self.docs).difference(*doc_list))
            term_freqs = [[0 for _ in range(len(doc_ids_matched))]]
            matched = (doc_ids_matched, term_freqs, term_freqs.copy())

        return matched

    def phrase_statement(
        self,
        docs_per_token: list[tuple[int, ...]],
        doc_pos_offset_per_token: list[tuple[int]],
        term_freqs: Sequence[Sequence[int]],
        term_freqs_title: Sequence[Sequence[int]],
    ) -> tuple[Sequence[int], Sequence[Sequence[int]], Sequence[Sequence[int]]]:
        if len(docs_per_token) == 1:
            return docs_per_token[0], term_freqs, term_freqs_title

        matched: list[int] = []
        (
            match_candidates,
            term_freqs_per_doc,
            term_freqs_title_per_doc,
            pos_tokens_per_doc_candidate,
        ) = self.intersection_phrase(
            docs_per_token, doc_pos_offset_per_token, term_freqs, term_freqs_title
        )

        if len(match_candidates) == 0:
            return matched, term_freqs_per_doc, term_freqs_title_per_doc

        pos_list_tokens_per_doc: list[list[tuple[int]]] = []
        pos_list_title_tokens_per_doc: list[list[tuple[int]]] = []
        for doc_idx, pos_offset_tuple in enumerate(pos_tokens_per_doc_candidate):
            pos_list_token: list[tuple[int]] = []
            pos_list_title_token: list[tuple[int]] = []
            for token_idx, pos_offset in enumerate(pos_offset_tuple):
                length_pos_list = struct.unpack(
                    "I", self.mm_position_list[pos_offset : pos_offset + INT_SIZE]
                )[0]
                pos_list_title: tuple[int] = struct.unpack(
                    f"{term_freqs_title_per_doc[doc_idx][token_idx]}I",
                    self.mm_position_list[
                        pos_offset + INT_SIZE : pos_offset
                        + INT_SIZE
                        + term_freqs_title_per_doc[doc_idx][token_idx] * INT_SIZE
                    ],
                )

                pos_list: tuple[int] = struct.unpack(
                    f"{term_freqs_per_doc[doc_idx][token_idx]}I",
                    self.mm_position_list[
                        pos_offset
                        + INT_SIZE
                        + term_freqs_title_per_doc[doc_idx][token_idx] : pos_offset
                        + INT_SIZE
                        + length_pos_list * INT_SIZE
                    ],
                )
                pos_list_title_token.append(pos_list_title)
                pos_list_token.append(pos_list)

            pos_list_title_tokens_per_doc.append(pos_list_title_token)
            pos_list_tokens_per_doc.append(pos_list_token)

        term_freqs_per_doc_matched = []
        term_freqs_title_per_doc_matched = []
        for doc_id, pos_list_per_token, term_freqs_doc, term_freqs_title_doc in zip(
            match_candidates,
            pos_list_tokens_per_doc,
            term_freqs_per_doc,
            term_freqs_title_per_doc,
        ):
            if self.has_phrase(
                pos_list_per_token,
            ):
                matched.append(doc_id)
                term_freqs_per_doc_matched.append(term_freqs_doc)
                term_freqs_title_per_doc_matched.append(term_freqs_title_doc)

        return (matched, term_freqs_per_doc_matched, term_freqs_title_per_doc_matched)

    def evaluate_subtree(
        self, node
    ) -> tuple[
        list[int], Sequence[int], Sequence[Sequence[int]], Sequence[Sequence[int]]
    ]:
        if isinstance(node.value, SearchMode):
            if node.value == SearchMode.AND:
                (
                    left_result_doc_freq,
                    left_result_doc_list,
                    left_result_term_freq,
                    left_result_term_freq_title,
                ) = self.evaluate_subtree(node.left)
                (
                    right_result_doc_freq,
                    right_result_doc_list,
                    right_result_term_freq,
                    right_result_term_freq_title,
                ) = self.evaluate_subtree(node.right)
                result_term_freq = list(left_result_term_freq)
                result_term_freq_title = list(left_result_term_freq_title)
                if (
                    not isinstance(node.left.value, str)
                    and not node.left.value == SearchMode.NOT
                ):
                    result_term_freq = [result_term_freq]
                    result_term_freq_title = [result_term_freq_title]
                if (
                    not isinstance(node.right.value, str)
                    and not node.right.value == SearchMode.NOT
                ):
                    right_result_term_freq = [right_result_term_freq]
                    right_result_term_freq_title = [right_result_term_freq_title]

                result_term_freq.extend(right_result_term_freq)  # pyright: ignore
                result_term_freq_title.extend(right_result_term_freq_title)  # pyright: ignore

                result_doc_freq = list(left_result_doc_freq)
                result_doc_freq.extend(right_result_doc_freq)
                return (
                    result_doc_freq,
                    *self.and_statement(
                        [left_result_doc_list, right_result_doc_list],
                        result_term_freq,  # pyright: ignore
                        result_term_freq_title,  # pyright: ignore
                    ),
                )
            elif node.value == SearchMode.OR:
                (
                    left_result_doc_freq,
                    left_result_doc_list,
                    left_result_term_freq,
                    left_result_term_freq_title,
                ) = self.evaluate_subtree(node.left)
                (
                    right_result_doc_freq,
                    right_result_doc_list,
                    right_result_term_freq,
                    right_result_term_freq_title,
                ) = self.evaluate_subtree(node.right)
                result_term_freq = list(left_result_term_freq)
                result_term_freq.extend(right_result_term_freq)

                result_term_freq_title = list(left_result_term_freq_title)
                result_term_freq_title.extend(right_result_term_freq_title)

                result_doc_freq = list(left_result_doc_freq)
                result_doc_freq.extend(right_result_doc_freq)
                return (
                    result_doc_freq,
                    *self.or_statement(
                        [left_result_doc_list, right_result_doc_list],
                        result_term_freq,
                        result_term_freq_title,
                    ),
                )
            elif node.value == SearchMode.NOT:
                (
                    left_result_doc_freq,
                    left_result_doc_list,
                    result_term_freq,
                    result_term_freq_title,
                ) = self.evaluate_subtree(node.left)
                return (
                    left_result_doc_freq,
                    *self.not_statement([left_result_doc_list]),
                )

        if isinstance(node.value, list):
            # phrase search
            tokens = node.value
            doc_list_phrase: list[tuple[int]] = []
            pos_offset_list_phrase: list[tuple[int]] = []
            term_freqs_phrase: list[tuple[int]] = []
            term_freqs_title_phrase: list[tuple[int]] = []
            doc_freqs_phrase: list[int] = []
            for token in tokens:
                (
                    doc_list_per_token,
                    pos_offset_list_per_token,
                    term_freq_per_token,
                    term_freq_title_per_token,
                ) = self.get_docs_phrase(token)
                doc_list_phrase.append(doc_list_per_token)
                pos_offset_list_phrase.append(pos_offset_list_per_token)
                term_freqs_phrase.append(term_freq_per_token)
                term_freqs_title_phrase.append(term_freq_title_per_token)
                doc_freqs_phrase.append(len(doc_list_per_token))
            return (
                doc_freqs_phrase,
                *self.phrase_statement(
                    doc_list_phrase,
                    pos_offset_list_phrase,
                    term_freqs_phrase,
                    term_freqs_title_phrase,
                ),
            )

        if isinstance(node.value, str):
            doc_list, term_frequencies, term_frequencies_title = self.get_docs(
                node.value
            )
            return (
                [len(doc_list)],
                doc_list,
                [list(term_frequencies)],
                [list(term_frequencies_title)],
            )

        return tuple([])

    def query_evaluator(
        self, tokens: list[str]
    ) -> tuple[
        list[int], Sequence[int], Sequence[Sequence[int]], Sequence[Sequence[int]]
    ]:
        output_queue = shunting_yard(tokens)
        root = build_query_tree(output_queue)
        doc_freqs, matched_doc_ids, matched_term_freqs, matched_term_freqs_title = (
            self.evaluate_subtree(root)
        )
        return doc_freqs, matched_doc_ids, matched_term_freqs, matched_term_freqs_title

    # Returns [0] the number of documents in which this token occurs, and [1] the number of bytes needed to skip to the
    # doc_list
    def get_doc_frequency_for_token(
        self,
        token: str
    ) -> tuple[int, int]:
        doc_id_file_offset: Optional[int] = self.index.get(token, None)
        if doc_id_file_offset is None:
            return 0, 0

        length_term: int = get_length_from_bytes(self.mm_doc_id_list, doc_id_file_offset)
        doc_id_file_offset += INT_SIZE + length_term
        length_doc_list: int = get_length_from_bytes(self.mm_doc_id_list, doc_id_file_offset)
        return length_doc_list, INT_SIZE + length_term + INT_SIZE

    def get_tokens_for_trigram(
        self,
        trigram: str
    ) -> set[tuple[str, int]]:
        result: set[tuple[str, int]] = set()

        byte_offset: int = self.trigram_to_tokens[trigram]
        num_tokens: int = get_length_from_bytes(self.mm_trigrams, byte_offset)
        byte_offset += INT_SIZE
        for token_id in range(num_tokens):
            token_length: int = get_length_from_bytes(self.mm_trigrams, byte_offset)
            byte_offset += INT_SIZE
            token: str = self.mm_trigrams[byte_offset : byte_offset + token_length].decode("utf-8")
            byte_offset += token_length
            num_trigrams_in_token: int = get_length_from_bytes(self.mm_trigrams, byte_offset)
            byte_offset += INT_SIZE

            result.add((token, num_trigrams_in_token))

        return result

    def correct_spelling(
        self,
        original_token: str,
        search_space_size_jaccard: int,
        search_space_size_edit_distance: int,
        num_replacements: int
    ) -> list[str]:
        trigrams = get_trigrams_from_token(original_token)
        num_trigrams: int = len(trigrams)
        if num_trigrams == 0:
            return [original_token]

        possible_replacements: list[set[tuple[str, int]]] = []
        for trigram in trigrams:
            possible_replacements.append(self.get_tokens_for_trigram(trigram))

        all_tokens: list[tuple[str, int]] = list(possible_replacements[0].union(*possible_replacements[1:]))
        jaccard_similarities: list[tuple[float, int]] = []
        for i, token in enumerate(all_tokens):
            overlap_size: int = 0
            for possible_replacement_set in possible_replacements:
                overlap_size += token in possible_replacement_set

            jaccard_similarity: float = overlap_size / (token[1] + num_trigrams - overlap_size)
            jaccard_similarities.append((jaccard_similarity, i))

        jaccard_similarities.sort(reverse=True)
        edit_distances: list[tuple[int, int]] = []
        for _, token_id in jaccard_similarities[:search_space_size_jaccard]:
            edit_distance = editdistance.eval(all_tokens[token_id][0], original_token)
            edit_distances.append((edit_distance, token_id))

        edit_distances.sort(reverse=False)
        document_frequencies: list[tuple[int, int]] = []
        for _, token_id in edit_distances[:search_space_size_edit_distance]:
            doc_frequency, _ = self.get_doc_frequency_for_token(all_tokens[token_id][0])
            document_frequencies.append((doc_frequency, token_id))

        document_frequencies.sort(reverse=True)
        result: list[str] = []
        for _, token_id in document_frequencies[:num_replacements]:
            result.append(all_tokens[token_id][0])

        print("Correcting spelling of \"" + original_token + "\" with \"" + result[0] + "\". Other possibilities: ", end="")
        first: bool = True
        for correction in result[1:]:
            if first:
                first = False
            else:
                print(", ", end="")
            print("\"" + correction + "\"", end="")
        print()

        return result

    def query_index_with_spelling_correction(
        self,
        token: str
    ) -> tuple[int, str]:
        res: Optional[int] = self.index.get(token, None)

        if res is None:
            token = self.correct_spelling(token, 75, 50, 5)[0]
            res = self.index.get(token, None)

        return res, token

    def get_docs(
        self,
        token: str,
        idf_threshold: float = 1.5,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        doc_id_file_offset, token = self.query_index_with_spelling_correction(token)

        if doc_id_file_offset is not None:
            length_doc_list, index_offset = self.get_doc_frequency_for_token(token)
            doc_id_file_offset += index_offset

            idf_score: float = self.calculate_idf(self.metadata["num_docs"], length_doc_list)

            if idf_score < idf_threshold or length_doc_list == 0:
                token = self.correct_spelling(token, 75, 50, 5)[0]
                length_doc_list, index_offset = self.get_doc_frequency_for_token(token)
                doc_id_file_offset += index_offset
                idf_score = self.calculate_idf(self.metadata["num_docs"], length_doc_list)

            if idf_score < idf_threshold or length_doc_list == 0:
                empty_tuple: tuple[int] = tuple([])
                return empty_tuple, empty_tuple, empty_tuple

            doc_list = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    doc_id_file_offset : doc_id_file_offset + length_doc_list * INT_SIZE
                ],  # + 4 and * 4 because we are on bytes level, but we use uint32 which is 4 bytes
            )
            term_frequencies_title = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    doc_id_file_offset + length_doc_list * INT_SIZE : doc_id_file_offset
                    + length_doc_list * INT_SIZE * 2
                ],
            )
            term_frequencies = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    doc_id_file_offset + length_doc_list * INT_SIZE * 2 : doc_id_file_offset
                    + length_doc_list
                    * INT_SIZE
                    * 3  # move to term frequency list: times 2 because we have to skip doc id list and title term frequency list
                ],
            )

            return doc_list, term_frequencies, term_frequencies_title
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            empty_tuple: tuple[int] = tuple([])
            return empty_tuple, empty_tuple, empty_tuple

    def get_docs_phrase(
        self,
        token: str,
    ) -> tuple[tuple[int], tuple[int], tuple[int], tuple[int]]:
        res: Optional[int] = self.query_index_with_spelling_correction(token)
        if res is not None:
            length_term: int = get_length_from_bytes(self.mm_doc_id_list, res)
            res += INT_SIZE + length_term  # move to the document list
            length_doc_list: int = get_length_from_bytes(self.mm_doc_id_list, res)
            doc_list = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    res + INT_SIZE : res + INT_SIZE + length_doc_list * INT_SIZE
                ],  # + 4 and * 4 because we are on bytes level, but we use uint32 which is 4 bytes
            )
            term_frequencies_title = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE : res
                    + INT_SIZE
                    + length_doc_list * INT_SIZE * 2
                ],
            )
            term_frequencies = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE * 2 : res
                    + INT_SIZE
                    + length_doc_list
                    * INT_SIZE
                    * 3  # move to term frequency list: times 2 because we have to skip doc id list and title term frequency list
                ],
            )

            pos_offset_list: tuple[int] = struct.unpack(
                f"{length_doc_list}Q",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE * 3 : res
                    + INT_SIZE
                    + length_doc_list
                    * INT_SIZE
                    * 3  # move to position offset list: times 2 because we have to skip doc id list and term frequency list
                    + length_doc_list * LONG_SIZE
                ],
            )

            return doc_list, pos_offset_list, term_frequencies, term_frequencies_title
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            empty_tuple: tuple[int] = tuple([])
            return empty_tuple, empty_tuple, empty_tuple, empty_tuple

    def get_doc_info(self, doc_id: int, snippet_length: int) -> DocumentInfo:
        offset = self.docs[doc_id]
        if len(self.docs) == doc_id + 1:
            next_offset = self.mm_doc_info.size()
        else:
            next_offset = self.docs[doc_id + 1]
        line = self.mm_doc_info[offset:next_offset].decode("utf-8") # TODO: THERE IS A BUG WHERE WE GET EMPTY LINES

        first_tab = line.index('\t')
        original_docid = line[:first_tab]
        second_tab = line.index('\t', first_tab + 1)
        url = line[first_tab + 1 : second_tab]
        title = line[second_tab + 1 : ]

        offset_in_offsets = doc_id * 8
        bodies_offset = struct.unpack("Q", self.mm_bodies_offsets[offset_in_offsets : offset_in_offsets + 8])[0] # the offsets stored here are 8 bytes
        bodies_end_offset = min(struct.unpack("Q", self.mm_bodies_offsets[offset_in_offsets + 8 : offset_in_offsets + 16])[0], bodies_offset + snippet_length)

        return DocumentInfo(
            original_docid=original_docid,
            url=url,
            title=title,
            body_snippet=self.mm_bodies[bodies_offset:bodies_end_offset].decode("utf-8", errors="replace")
        )

    def calculate_idf(self, N: int, doc_freq: int) -> float:
        return math.log((N - doc_freq + 0.5) / (doc_freq + 0.5))

    def calculate_term_weight(
        self,
        tf: int,
        doc_length: int,
        avg_length: float,
        b: float = 0.75,
    ) -> float:
        return tf / (1 - b + b * (doc_length / avg_length))

    def fielded_bm25_score(
        self,
        idf_tokens: list[float],
        tf_tokens: list[float],
        k: float = 1.6,
    ) -> float:
        score = 0
        for idf_token, tf_token in zip(idf_tokens, tf_tokens):
            score += idf_token * (tf_token * (k + 1)) / (tf_token + k)
        return score

    def bm25_score(
        self,
        df_tokens: list[int],
        tf_tokens: Sequence[int],
        N: int,
        doc_length: int,
        avg_length: float,
    ) -> float:
        k = 1.6
        b = 0.75
        score = 0
        for df, tf in zip(df_tokens, tf_tokens):
            idf = self.calculate_idf(N, df)
            bracket_term = (
                tf * (k + 1) / (tf + k * (1 - b + b * (doc_length / avg_length)))
            )
            score += idf * bracket_term

        return score

    def search(
        self, query: str, mode: SearchMode, num_return: int = 10, snippet_length: int = 100
    ) -> tuple[int, list[tuple[float, DocumentInfo]]]:
        tokens = tokenize_text(query)

        doc_list: list[tuple[int, ...]] = []
        term_freqs: list[tuple[int, ...]] = []
        term_freqs_title: list[tuple[int, ...]] = []
        doc_freqs: list[int] = []
        match mode:
            case SearchMode.PHRASE:
                pos_offset_list: list[tuple[int]] = []
                for token in tokens:
                    (
                        doc_list_per_token,
                        pos_offset_list_per_token,
                        term_freq_per_token,
                        term_freq_title_per_token,
                    ) = self.get_docs_phrase(token)
                    doc_list.append(doc_list_per_token)
                    pos_offset_list.append(pos_offset_list_per_token)
                    term_freqs.append(term_freq_per_token)
                    term_freqs_title.append(term_freq_title_per_token)
                    doc_freqs.append(len(doc_list_per_token))
            case SearchMode.AND | SearchMode.OR | SearchMode.NOT:
                for token in tokens:
                    (
                        doc_list_per_token,
                        term_freq_per_token,
                        term_freq_title_per_token,
                    ) = self.get_docs(token)
                    if len(doc_list_per_token) > 0:
                        doc_list.append(doc_list_per_token)
                        term_freqs.append(term_freq_per_token)
                        term_freqs_title.append(term_freq_title_per_token)
                        doc_freqs.append(len(doc_list_per_token))
            case SearchMode.QUERY_EVALUATOR:
                pass
            case _:
                raise ValueError(f"Unsupported search mode: {mode}")

        matched_doc_ids: Sequence[int] = []
        matched_term_freqs: Sequence[Sequence[int]] = []
        if mode == SearchMode.AND:
            matched_doc_ids, matched_term_freqs, matched_term_freqs_title = (
                self.and_statement(doc_list, term_freqs, term_freqs_title)
            )
        elif mode == SearchMode.OR:
            matched_doc_ids, matched_term_freqs, matched_term_freqs_title = (
                self.or_statement(doc_list, term_freqs, term_freqs_title)
            )
        elif mode == SearchMode.NOT:
            matched_doc_ids, matched_term_freqs, matched_term_freqs_title = (
                self.not_statement(doc_list)
            )
        elif mode == SearchMode.PHRASE:
            matched_doc_ids, matched_term_freqs, matched_term_freqs_title = (
                self.phrase_statement(
                    doc_list, pos_offset_list, term_freqs, term_freqs_title
                )
            )
        elif mode == SearchMode.QUERY_EVALUATOR:
            doc_freqs, matched_doc_ids, matched_term_freqs, matched_term_freqs_title = (
                self.query_evaluator(tokens)
            )

        def flatten(items: Sequence[Sequence[int] | int]) -> list[int]:
            flat_list: list[int] = []
            for item in items:
                if isinstance(item, int):
                    flat_list.append(item)
                else:
                    flattened_sublist = flatten(item)
                    flat_list.extend(flattened_sublist)
            return flat_list

        result_candidates: list[tuple[float, int]] = []

        if len(matched_term_freqs) == 1 and len(matched_doc_ids) != 1:
            matched_term_freqs = list(zip(*matched_term_freqs))

        assert len(matched_doc_ids) == len(matched_term_freqs)

        idf_per_token = [
            self.calculate_idf(self.metadata["num_docs"], df) for df in doc_freqs
        ]

        avg_doc_length = self.metadata["average_doc_length"]
        calculate_term_weight_body = partial(
            self.calculate_term_weight, avg_length=avg_doc_length
        )
        avg_title_length = self.metadata["average_title_length"]
        calculate_term_weight_title = partial(
            self.calculate_term_weight, avg_length=avg_title_length
        )

        for doc_id, term_freqs_token, term_freqs_token_title in zip(
            matched_doc_ids, matched_term_freqs, matched_term_freqs_title
        ):
            term_freqs_token = flatten(term_freqs_token)
            term_freqs_token_title = flatten(term_freqs_token_title)

            doc_length = self.document_lengths[doc_id]
            tilte_length = self.title_lengths[doc_id]

            term_weights_body = [
                calculate_term_weight_body(
                    tf=tf,
                    doc_length=doc_length,
                )
                for tf in term_freqs_token
            ]

            term_weights_title = [
                calculate_term_weight_title(
                    tf=tf,
                    doc_length=tilte_length,
                )
                for tf in term_freqs_token_title
            ]

            weight_title = 2.0
            term_weights = [
                body_weight + title_weight * weight_title
                for body_weight, title_weight in zip(
                    term_weights_body, term_weights_title
                )
            ]

            score = self.fielded_bm25_score(
                idf_tokens=idf_per_token,
                tf_tokens=term_weights,
            )

            result_candidates.append((score, doc_id))

        results: list[tuple[float, DocumentInfo]] = []
        result_candidates = sorted(result_candidates, key=lambda x: x[0], reverse=True)[:num_return]

        for score, doc_id in result_candidates:
            results.append((score, self.get_doc_info(doc_id, snippet_length)))

        return len(matched_doc_ids), results
