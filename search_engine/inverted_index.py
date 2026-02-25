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
from typing import NamedTuple, Optional, Sequence

import editdistance
import marisa_trie
import numpy as np
import torch
from ordered_set import OrderedSet
from sentence_transformers import SentenceTransformer, util

from search_engine.preprocessing import build_query_tree, shunting_yard, tokenize_text
from search_engine.ranking_model.model import RankingModel
from search_engine.utils import (
    INT_SIZE,
    LONG_SIZE,
    DocumentInfo,
    SearchMode,
    get_length_from_bytes,
    get_trigrams_from_token,
)


class SemTradContainResult(NamedTuple):
    org_sem_idx: int
    doc_id: int
    trad_idx: Optional[int]


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
        file_path_trigram_offsets: str | Path,
        file_path_ranking_model: str | Path,
        file_path_embeddings: str | Path,
        file_path_embedding_metadata: str | Path,
        enable_semantic_search: bool = True,
        enable_spelling_correction: bool = True,
    ) -> None:
        doc_id_file = open(file_path_doc_id, "rb")

        position_list_file = open(file_path_position_list, mode="rb")
        bodies_file = open(file_path_bodies, mode="rb")
        bodies_offsets_file = open(file_path_bodies_offsets, mode="rb")
        trigrams_file = open(file_path_trigrams, mode="rb")

        doc_info_file = open(file_path_doc_info, "rb")

        self.index = marisa_trie.RecordTrie("<Q").mmap(str(file_path_term_index))
        self.enable_spelling_correction = enable_spelling_correction
        self.enable_semantic_search = enable_semantic_search

        checkpoint = torch.load(file_path_ranking_model, weights_only=False)
        self.ranking_model = RankingModel(
            input_dim=checkpoint["input_dim"], hidden_dim=checkpoint["hidden_dim"]
        )
        self.ranking_model.load_state_dict(checkpoint["model_state_dict"])
        self.ranking_model.eval()

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
        self.mm_bodies_offsets = mmap.mmap(
            bodies_offsets_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        with open(file_path_doc_info_offset, "rb") as f:
            self.docs: array[int] = array("I")
            if self.docs.itemsize != 4:
                self.docs = array("L")
                if self.docs.itemsize != 4:
                    raise RuntimeError(
                        "Machine does not have an exact 4 byte integer type"
                    )
            file_bytes: int = os.path.getsize(file_path_doc_info_offset)
            self.docs.fromfile(f, file_bytes // 4)

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

        self.mm_trigrams = mmap.mmap(
            trigrams_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        with open(file_path_metadata, "rb") as f:
            self.metadata = pickle.load(f)

        with open(file_path_document_lengths, "rb") as f:
            file_bytes: int = os.path.getsize(file_path_document_lengths)
            self.document_lengths: array = array("I")
            if self.document_lengths.itemsize != 4:
                self.document_lengths = array("L")
                if self.document_lengths.itemsize != 4:
                    raise RuntimeError(
                        "Machine does not have an exact 4 byte integer type"
                    )
            self.document_lengths.fromfile(f, file_bytes // 4)  # uint32_t

        with open(file_path_title_lengths, "rb") as f:
            file_bytes: int = os.path.getsize(file_path_document_lengths)
            self.title_lengths: array = array("I")
            if self.title_lengths.itemsize != 4:
                self.title_lengths = array("L")
                if self.title_lengths.itemsize != 4:
                    raise RuntimeError(
                        "Machine does not have an exact 4 byte integer type"
                    )
            self.title_lengths.fromfile(f, file_bytes // 4)  # uint32_t

        avg_doc_length = self.metadata["average_doc_length"]

        self.calculate_term_weight_body = partial(
            self.calculate_term_weight, avg_length=avg_doc_length
        )
        avg_title_length = self.metadata["average_title_length"]

        self.calculate_term_weight_title = partial(
            self.calculate_term_weight, avg_length=avg_title_length
        )

        if enable_semantic_search:
            with open(file_path_embedding_metadata, "rb") as f:
                self.embedding_metadata = pickle.load(f)

            with open(file_path_embeddings, "rb") as f:
                self.doc_embeddings = np.reshape(
                    np.fromfile(f, dtype=np.float32),
                    shape=(
                        self.embedding_metadata["num_docs"],
                        self.embedding_metadata["truncate_dim"],
                    ),
                )

            self.sentence_transformer = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="mps" if torch.backends.mps.is_available() else "cpu",
            )
            self.sentence_transformer.max_seq_length = 256

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
        pos_offset_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]], list[list[int]]]:
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
                    pos_offset_list[i][0],
                    term_frequencies[i][0],
                    term_frequencies_title[i][0],
                ),
            )

        counter_same_value = 0
        last_min = -1

        result_pos_offsets: list[list[int]] = []
        last_pos_offsets: list[int] = []

        result_term_freqs: list[list[int]] = []
        last_term_freqs: list[int] = []

        result_term_freqs_title: list[list[int]] = []
        last_term_freqs_title: list[int] = []

        one_list_finished = False

        while min_heap:
            (
                current_min,
                current_index,
                pos_offsets,
                term_frequency,
                term_frequency_title,
            ) = heapq.heappop(min_heap)

            if last_min == current_min:
                counter_same_value += 1
            else:
                if one_list_finished:
                    break
                last_pos_offsets = []
                last_term_freqs = []
                last_term_freqs_title = []
                counter_same_value = 0

            last_pos_offsets.append(pos_offsets)
            last_term_freqs.append(term_frequency)
            last_term_freqs_title.append(term_frequency_title)

            if counter_same_value == len(doc_ids) - 1:
                result_pos_offsets.append(last_pos_offsets)
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
                    pos_offset_list[current_index][pointer[current_index]],
                    term_frequencies[current_index][pointer[current_index]],
                    term_frequencies_title[current_index][pointer[current_index]],
                ),
            )
            last_min = current_min
        return (
            result_doc_ids,
            result_pos_offsets,
            result_term_freqs,
            result_term_freqs_title,
        )

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
        pos_offset_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]], list[list[int]]]:
        num_terms = len(doc_ids)
        pointers = [0] * num_terms
        result_doc_ids = []
        result_pos_offsets = []
        result_term_freqs = []
        result_term_freqs_title = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids):
            if doc_list:
                heapq.heappush(min_heap, (doc_list[0], i))

        last_doc_id = -1
        current_pos_offset_vector = []
        current_tf_vector = []
        current_tf_vector_title = []
        while min_heap:
            doc_id, term_index = heapq.heappop(min_heap)
            if doc_id != last_doc_id:
                if last_doc_id != -1:
                    result_pos_offsets.append(current_pos_offset_vector)
                    result_term_freqs.append(current_tf_vector)
                    result_term_freqs_title.append(current_tf_vector_title)

                last_doc_id = doc_id
                result_doc_ids.append(doc_id)
                current_pos_offset_vector = [0] * num_terms
                current_tf_vector = [0] * num_terms
                current_tf_vector_title = [0] * num_terms

            pos_offset = pos_offset_list[term_index][pointers[term_index]]
            tf = term_frequencies[term_index][pointers[term_index]]
            tf_title = term_frequencies_title[term_index][pointers[term_index]]
            current_pos_offset_vector[term_index] = pos_offset
            current_tf_vector[term_index] = tf
            current_tf_vector_title[term_index] = tf_title
            pointers[term_index] += 1

            if pointers[term_index] < len(doc_ids[term_index]):
                next_doc_id = doc_ids[term_index][pointers[term_index]]
                heapq.heappush(min_heap, (next_doc_id, term_index))

        if last_doc_id != -1:
            result_pos_offsets.append(current_pos_offset_vector)
            result_term_freqs.append(current_tf_vector)
            result_term_freqs_title.append(current_tf_vector_title)

        return (
            result_doc_ids,
            result_pos_offsets,
            result_term_freqs,
            result_term_freqs_title,
        )

    def and_statement(
        self,
        doc_list: Sequence[Sequence[int]],
        pos_offset_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[
        Sequence[int],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
    ]:
        matched: tuple[
            Sequence[int],
            Sequence[Sequence[int]],
            Sequence[Sequence[int]],
            Sequence[Sequence[int]],
        ] = [], [], [], []
        if len(doc_list) == 1:
            matched = (
                doc_list[0],
                list(zip(*pos_offset_list)),
                list(zip(*term_frequencies)),
                list(zip(*term_frequencies_title)),
            )
        elif len(doc_list) > 1:
            matched = self.intersection(
                doc_list, pos_offset_list, term_frequencies, term_frequencies_title
            )

        return matched

    def or_statement(
        self,
        doc_list: Sequence[Sequence[int]],
        pos_offset_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
        term_frequencies_title: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], list[list[int]], list[list[int]]]:
        matched: tuple[list[int], list[list[int]], list[list[int]], list[list[int]]] = (
            [],
            [],
            [],
            [],
        )

        if len(doc_list) == 1:
            matched = (
                list(doc_list[0]),
                [list(pos_offset_list[0])],
                [list(term_frequencies[0])],
                [list(term_frequencies_title[0])],
            )
        elif len(doc_list) > 1:
            matched = self.union(
                doc_list, pos_offset_list, term_frequencies, term_frequencies_title
            )

        return matched

    def not_statement(
        self, doc_list: Sequence[Sequence[int]], *args
    ) -> tuple[list[int], list[list[int]], list[list[int]], list[list[int]]]:
        matched: tuple[list[int], list[list[int]], list[list[int]], list[list[int]]] = (
            tuple([])
        )
        if len(doc_list) == 0:
            term_freqs = [[0 for _ in range(len(self.docs))]]
            pos_offsets = [[-1 for _ in range(len(self.docs))]]
            matched = (list(self.docs)), pos_offsets, term_freqs, term_freqs.copy()
        else:
            doc_ids_matched = list(OrderedSet(self.docs).difference(*doc_list))
            term_freqs = [[0 for _ in range(len(doc_ids_matched))]]
            pos_offsets = [[-1 for _ in range(len(self.docs))]]
            matched = (doc_ids_matched, pos_offsets, term_freqs, term_freqs.copy())

        return matched

    def get_pos_offsets(
        self, pos_tokens_per_doc_candidate, term_freqs_per_doc, term_freqs_title_per_doc
    ):
        pos_list_tokens_per_doc: list[list[tuple[int]]] = []
        pos_list_title_tokens_per_doc: list[list[tuple[int]]] = []
        for doc_idx, pos_offset_tuple in enumerate(pos_tokens_per_doc_candidate):
            pos_list_token: list[tuple[int]] = []
            pos_list_title_token: list[tuple[int]] = []
            for token_idx, pos_offset in enumerate(pos_offset_tuple):
                if (
                    term_freqs_per_doc[doc_idx][token_idx] == 0
                    and term_freqs_title_per_doc[doc_idx][token_idx] == 0
                ):
                    pos_list_token.append(tuple())
                    pos_list_title_token.append(tuple())
                    continue
                else:
                    length_pos_list = struct.unpack(
                        "I", self.mm_position_list[pos_offset : pos_offset + INT_SIZE]
                    )[0]

                if term_freqs_per_doc[doc_idx][token_idx] == 0:
                    pos_list_token.append(tuple())
                else:
                    pos_list: tuple[int] = struct.unpack(
                        f"{term_freqs_per_doc[doc_idx][token_idx]}I",
                        self.mm_position_list[
                            pos_offset
                            + INT_SIZE
                            + term_freqs_title_per_doc[doc_idx][token_idx]
                            * INT_SIZE : pos_offset
                            + INT_SIZE
                            + length_pos_list * INT_SIZE
                        ],
                    )
                    pos_list_token.append(pos_list)

                if term_freqs_title_per_doc[doc_idx][token_idx] == 0:
                    pos_list_title_token.append(tuple())
                else:
                    pos_list_title: tuple[int] = struct.unpack(
                        f"{term_freqs_title_per_doc[doc_idx][token_idx]}I",
                        self.mm_position_list[
                            pos_offset + INT_SIZE : pos_offset
                            + INT_SIZE
                            + term_freqs_title_per_doc[doc_idx][token_idx] * INT_SIZE
                        ],
                    )
                    pos_list_title_token.append(pos_list_title)

            pos_list_title_tokens_per_doc.append(pos_list_title_token)
            pos_list_tokens_per_doc.append(pos_list_token)
        return pos_list_tokens_per_doc, pos_list_title_tokens_per_doc

    def phrase_statement(
        self,
        docs_per_token: list[tuple[int, ...]],
        doc_pos_offset_per_token: list[tuple[int, ...]],
        term_freqs: Sequence[Sequence[int]],
        term_freqs_title: Sequence[Sequence[int]],
    ) -> tuple[
        Sequence[int],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
    ]:
        if len(docs_per_token) == 1:
            return (
                docs_per_token[0],
                doc_pos_offset_per_token,
                term_freqs,
                term_freqs_title,
            )

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
            return (
                matched,
                pos_tokens_per_doc_candidate,
                term_freqs_per_doc,
                term_freqs_title_per_doc,
            )

        pos_list_tokens_per_doc, pos_list_title_tokens_per_doc = self.get_pos_offsets(
            pos_tokens_per_doc_candidate, term_freqs_per_doc, term_freqs_title_per_doc
        )

        term_freqs_per_doc_matched = []
        term_freqs_title_per_doc_matched = []
        pos_offsets_per_doc_matched = []
        for (
            doc_id,
            pos_list_per_token,
            pos_list_title_per_token,
            term_freqs_doc,
            term_freqs_title_doc,
            pos_offsets,
        ) in zip(
            match_candidates,
            pos_list_tokens_per_doc,
            pos_list_title_tokens_per_doc,
            term_freqs_per_doc,
            term_freqs_title_per_doc,
            pos_tokens_per_doc_candidate,
        ):
            if all(pos_list for pos_list in pos_list_per_token) and self.has_phrase(
                pos_list_per_token
            ):
                matched.append(doc_id)
                pos_offsets_per_doc_matched.append(pos_offsets)
                term_freqs_per_doc_matched.append(term_freqs_doc)
                term_freqs_title_per_doc_matched.append(term_freqs_title_doc)
            elif all(
                pos_list for pos_list in pos_list_title_per_token
            ) and self.has_phrase(pos_list_title_per_token):
                matched.append(doc_id)
                pos_offsets_per_doc_matched.append(pos_offsets)
                term_freqs_per_doc_matched.append(term_freqs_doc)
                term_freqs_title_per_doc_matched.append(term_freqs_title_doc)

        return (
            matched,
            pos_offsets_per_doc_matched,
            term_freqs_per_doc_matched,
            term_freqs_title_per_doc_matched,
        )

    def evaluate_subtree(
        self, node
    ) -> tuple[
        list[int],
        Sequence[int],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
    ]:
        if isinstance(node.value, SearchMode):
            if node.value == SearchMode.AND:
                (
                    left_result_doc_freq,
                    left_result_doc_list,
                    left_result_pos_offset,
                    left_result_term_freq,
                    left_result_term_freq_title,
                ) = self.evaluate_subtree(node.left)
                (
                    right_result_doc_freq,
                    right_result_doc_list,
                    right_result_pos_offset,
                    right_result_term_freq,
                    right_result_term_freq_title,
                ) = self.evaluate_subtree(node.right)
                result_pos_offsets = list(left_result_pos_offset)
                result_term_freq = list(left_result_term_freq)
                result_term_freq_title = list(left_result_term_freq_title)
                if (
                    not isinstance(node.left.value, str)
                    and not node.left.value == SearchMode.NOT
                ):
                    result_pos_offsets = [result_pos_offsets]
                    result_term_freq = [result_term_freq]
                    result_term_freq_title = [result_term_freq_title]
                if (
                    not isinstance(node.right.value, str)
                    and not node.right.value == SearchMode.NOT
                ):
                    right_result_pos_offset = [right_result_pos_offset]
                    right_result_term_freq = [right_result_term_freq]
                    right_result_term_freq_title = [right_result_term_freq_title]

                result_pos_offsets.extend(right_result_pos_offset)  # pyright: ignore
                result_term_freq.extend(right_result_term_freq)  # pyright: ignore
                result_term_freq_title.extend(right_result_term_freq_title)  # pyright: ignore

                result_doc_freq = list(left_result_doc_freq)
                result_doc_freq.extend(right_result_doc_freq)
                return (
                    result_doc_freq,
                    *self.and_statement(
                        [left_result_doc_list, right_result_doc_list],
                        result_pos_offsets,  # pyright: ignore
                        result_term_freq,  # pyright: ignore
                        result_term_freq_title,  # pyright: ignore
                    ),
                )
            elif node.value == SearchMode.OR:
                (
                    left_result_doc_freq,
                    left_result_doc_list,
                    left_result_pos_offset,
                    left_result_term_freq,
                    left_result_term_freq_title,
                ) = self.evaluate_subtree(node.left)
                (
                    right_result_doc_freq,
                    right_result_doc_list,
                    right_result_pos_offset,
                    right_result_term_freq,
                    right_result_term_freq_title,
                ) = self.evaluate_subtree(node.right)
                result_pos_offsets = list(left_result_pos_offset)
                result_pos_offsets.extend(right_result_pos_offset)

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
                        result_pos_offsets,
                        result_term_freq,
                        result_term_freq_title,
                    ),
                )
            elif node.value == SearchMode.NOT:
                (
                    left_result_doc_freq,
                    left_result_doc_list,
                    left_result_pos_offset,
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
            (
                doc_list,
                pos_offset_list_per_token,
                term_frequencies,
                term_frequencies_title,
            ) = self.get_docs(node.value)
            return (
                [len(doc_list)],
                doc_list,
                [list(pos_offset_list_per_token)],
                [list(term_frequencies)],
                [list(term_frequencies_title)],
            )

        return tuple([])

    def query_evaluator(
        self, tokens: list[str]
    ) -> tuple[
        list[int],
        Sequence[int],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
        Sequence[Sequence[int]],
    ]:
        output_queue = shunting_yard(tokens)
        root = build_query_tree(output_queue)
        (
            doc_freqs,
            matched_doc_ids,
            matched_pos_offsets,
            matched_term_freqs,
            matched_term_freqs_title,
        ) = self.evaluate_subtree(root)
        return (
            doc_freqs,
            matched_doc_ids,
            matched_pos_offsets,
            matched_term_freqs,
            matched_term_freqs_title,
        )

    # Returns [0] the number of documents in which this token occurs, and [1] the number of bytes needed to skip to the
    # doc_list
    def get_doc_frequency_for_token(self, token: str) -> tuple[int, int]:
        result: list[list[int]] = self.index.get(token)
        doc_id_file_offset: Optional[int] = result[0][0] if result else None

        if doc_id_file_offset is None:
            return 0, 0

        length_term: int = get_length_from_bytes(
            self.mm_doc_id_list, doc_id_file_offset
        )
        doc_id_file_offset += INT_SIZE + length_term
        length_doc_list: int = get_length_from_bytes(
            self.mm_doc_id_list, doc_id_file_offset
        )
        return length_doc_list, INT_SIZE + length_term + INT_SIZE

    def get_tokens_for_trigram(self, trigram: str) -> set[tuple[str, int]]:
        result: set[tuple[str, int]] = set()

        byte_offset: int = self.trigram_to_tokens[trigram]
        num_tokens: int = get_length_from_bytes(self.mm_trigrams, byte_offset)
        byte_offset += INT_SIZE
        for token_id in range(num_tokens):
            token_length: int = get_length_from_bytes(self.mm_trigrams, byte_offset)
            byte_offset += INT_SIZE
            token: str = self.mm_trigrams[
                byte_offset : byte_offset + token_length
            ].decode("utf-8")
            byte_offset += token_length
            num_trigrams_in_token: int = get_length_from_bytes(
                self.mm_trigrams, byte_offset
            )
            byte_offset += INT_SIZE

            result.add((token, num_trigrams_in_token))

        return result

    def correct_spelling(
        self,
        original_token: str,
        search_space_size_jaccard: int,
        search_space_size_edit_distance: int,
        num_replacements: int,
    ) -> list[str]:
        trigrams = get_trigrams_from_token(original_token)
        num_trigrams: int = len(trigrams)
        if num_trigrams == 0:
            return [original_token]

        possible_replacements: list[set[tuple[str, int]]] = []
        for trigram in trigrams:
            possible_replacements.append(self.get_tokens_for_trigram(trigram))

        all_tokens: list[tuple[str, int]] = list(
            possible_replacements[0].union(*possible_replacements[1:])
        )
        jaccard_similarities: list[tuple[float, int]] = []
        for i, token in enumerate(all_tokens):
            overlap_size: int = 0
            for possible_replacement_set in possible_replacements:
                overlap_size += token in possible_replacement_set

            jaccard_similarity: float = overlap_size / (
                token[1] + num_trigrams - overlap_size
            )
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

        print(
            f'Correcting spelling of "{original_token}" with "{result[0]}". Other possibilities: ',
            end="",
        )
        first: bool = True
        for correction in result[1:]:
            if first:
                first = False
            else:
                print(", ", end="")
            print(f'"{correction}"', end="")
        print()

        return result

    def query_index_with_spelling_correction(
        self, token: str
    ) -> tuple[int | None, str]:
        result: list[list[int]] = self.index.get(token)
        res: Optional[int] = result[0][0] if result else None

        if (res is None) and self.enable_spelling_correction:
            token = self.correct_spelling(token, 75, 50, 5)[0]
            result = self.index.get(token)
            res = result[0][0] if result else None

        return res, token

    def get_doc_list(self, length_doc_list: int, doc_id_file_offset: int):
        return struct.unpack(
            f"{length_doc_list}I",
            self.mm_doc_id_list[
                doc_id_file_offset : doc_id_file_offset + length_doc_list * INT_SIZE
            ],  # + 4 and * 4 because we are on bytes level, but we use uint32 which is 4 bytes
        )

    def get_term_frequencies(self, length_doc_list: int, doc_id_file_offset: int):
        return [
            struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    doc_id_file_offset + length_doc_list * INT_SIZE : doc_id_file_offset
                    + length_doc_list * INT_SIZE * 2
                ],
            ),
            struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    doc_id_file_offset
                    + length_doc_list * INT_SIZE * 2 : doc_id_file_offset
                    + length_doc_list
                    * INT_SIZE
                    * 3
                    # move to term frequency list: times 2 because we have to skip doc id list and title term frequency list
                ],
            ),
        ]

    def get_docs(
        self,
        token: str,
        idf_threshold: float = 1.5,
        enable_threshold: bool = True,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        doc_id_file_offset, token = self.query_index_with_spelling_correction(token)

        if doc_id_file_offset is not None:
            length_doc_list, index_offset = self.get_doc_frequency_for_token(token)
            doc_id_file_offset += index_offset

            idf_score: float = self.calculate_idf(
                self.metadata["num_docs"], length_doc_list
            )

            if (
                idf_score < idf_threshold or length_doc_list == 0
            ) and self.enable_spelling_correction:
                token = self.correct_spelling(token, 75, 50, 5)[0]
                length_doc_list, index_offset = self.get_doc_frequency_for_token(token)
                doc_id_file_offset += index_offset
                idf_score = self.calculate_idf(
                    self.metadata["num_docs"], length_doc_list
                )

            if (enable_threshold) and (
                idf_score < idf_threshold or length_doc_list == 0
            ):
                empty_tuple: tuple[int] = tuple([])
                return empty_tuple, empty_tuple, empty_tuple, empty_tuple

            doc_list = self.get_doc_list(length_doc_list, doc_id_file_offset)
            term_frequencies_title, term_frequencies = self.get_term_frequencies(
                length_doc_list, doc_id_file_offset
            )

            pos_offset_list: tuple[int] = struct.unpack(
                f"{length_doc_list}Q",
                self.mm_doc_id_list[
                    doc_id_file_offset
                    + length_doc_list * INT_SIZE * 3 : doc_id_file_offset
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

    def get_docs_phrase(
        self, token: str
    ) -> tuple[tuple[int], tuple[int], tuple[int], tuple[int]]:
        result: list[list[int]] = self.index.get(token)
        doc_id_file_offset: Optional[int] = result[0][0] if result else None

        if doc_id_file_offset is not None:
            length_doc_list, index_offset = self.get_doc_frequency_for_token(token)
            doc_id_file_offset += index_offset

            doc_list = self.get_doc_list(length_doc_list, doc_id_file_offset)
            term_frequencies_title, term_frequencies = self.get_term_frequencies(
                length_doc_list, doc_id_file_offset
            )

            pos_offset_list: tuple[int] = struct.unpack(
                f"{length_doc_list}Q",
                self.mm_doc_id_list[
                    doc_id_file_offset
                    + length_doc_list * INT_SIZE * 3 : doc_id_file_offset
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
        line = self.mm_doc_info[offset:next_offset].decode("utf-8")

        first_tab = line.index("\t")
        original_docid = line[:first_tab]
        second_tab = line.index("\t", first_tab + 1)
        url = line[first_tab + 1 : second_tab]
        title = line[second_tab + 1 :]

        offset_in_offsets = doc_id * 8
        bodies_offset = struct.unpack(
            "Q", self.mm_bodies_offsets[offset_in_offsets : offset_in_offsets + 8]
        )[0]  # the offsets stored here are 8 bytes
        bodies_end_offset = min(
            struct.unpack(
                "Q",
                self.mm_bodies_offsets[offset_in_offsets + 8 : offset_in_offsets + 16],
            )[0],
            bodies_offset + snippet_length,
        )

        return DocumentInfo(
            original_docid=original_docid,
            url=url,
            title=title,
            body_snippet=self.mm_bodies[bodies_offset:bodies_end_offset].decode(
                "utf-8", errors="replace"
            ),
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

    def extract_features(
        self,
        matched_doc_ids: Sequence[int],
        matched_bm25_scores: Sequence[float],
        matched_bm25_scores_body: Sequence[float],
        matched_bm25_scores_title: Sequence[float],
        matched_term_freqs: Sequence[Sequence[int]],
        matched_term_freqs_title: Sequence[Sequence[int]],
        matched_pos_offsets: Sequence[Sequence[int | None]],
    ):
        # ["bm25_score","bm25_score_body","bm25_score_title","body_first_occurrence_mean","title_first_occurrence_mean",
        # "body_first_occurrence_min","title_first_occurrence_min","body_length_norm","title_length_norm","in_title"]

        max_doc_length = self.metadata["max_doc_length"]
        max_title_length = self.metadata["max_title_length"]

        bm25_scores = []
        bm25_scores_body = []
        bm25_scores_title = []
        body_first_occurrence_mean = []
        title_first_occurrence_mean = []
        body_first_occurrence_min = []
        title_first_occurrence_min = []
        body_length_norm = []
        title_length_norm = []
        in_title = []

        matched_pos_offsets = [
            self.flatten(pos_offset_tuple) for pos_offset_tuple in matched_pos_offsets
        ]
        matched_term_freqs_flattened = [
            self.flatten(term_freq_tuple) for term_freq_tuple in matched_term_freqs
        ]
        matched_term_freqs_title_flattend = [
            self.flatten(term_freq_title_tuple)
            for term_freq_title_tuple in matched_term_freqs_title
        ]
        pos_list_tokens_per_doc, pos_list_title_tokens_per_doc = self.get_pos_offsets(
            matched_pos_offsets,
            matched_term_freqs_flattened,
            matched_term_freqs_title_flattend,
        )

        for (
            doc_id,
            term_freqs_token,
            term_freqs_token_title,
            bm25_score,
            bm25_score_body,
            bm25_score_title,
            pos_list_tokens,
            pos_list_tokens_title,
        ) in zip(
            matched_doc_ids,
            matched_term_freqs_flattened,
            matched_term_freqs_title_flattend,
            matched_bm25_scores,
            matched_bm25_scores_body,
            matched_bm25_scores_title,
            pos_list_tokens_per_doc,
            pos_list_title_tokens_per_doc,
        ):
            term_freqs_token = self.flatten(term_freqs_token)
            term_freqs_token_title = self.flatten(term_freqs_token_title)

            doc_length = self.document_lengths[doc_id]
            title_length = self.title_lengths[doc_id]

            bm25_scores.append(bm25_score)
            bm25_scores_body.append(bm25_score_body)
            bm25_scores_title.append(bm25_score_title)

            body_first_occurrence = [
                pos_list[0] / doc_length if len(pos_list) > 0 else 1.0
                for pos_list in pos_list_tokens
            ]
            title_first_occurrence = [
                pos_list[0] / title_length if len(pos_list) > 0 else 1.0
                for pos_list in pos_list_tokens_title
            ]

            body_first_occurrence_mean.append(
                sum(body_first_occurrence) / len(body_first_occurrence)
            )
            title_first_occurrence_mean.append(
                sum(title_first_occurrence) / len(title_first_occurrence)
            )

            body_first_occurrence_min.append(min(body_first_occurrence))
            title_first_occurrence_min.append(min(title_first_occurrence))

            body_length_norm.append(doc_length / max_doc_length)
            title_length_norm.append(title_length / max_title_length)

            in_title.append(int(any(idx < 1.0 for idx in title_first_occurrence)))

        return torch.tensor(
            [
                bm25_scores,
                bm25_scores_body,
                bm25_scores_title,
                body_first_occurrence_mean,
                title_first_occurrence_mean,
                body_first_occurrence_min,
                title_first_occurrence_min,
                body_length_norm,
                title_length_norm,
                in_title,
            ]
        ).T

    def flatten(
        self, items: Sequence[Sequence[Optional[int]] | Optional[int]]
    ) -> list[int | None]:
        flat_list: list[int | None] = []
        for item in items:
            if isinstance(item, int) or item is None:
                flat_list.append(item)
            else:
                flattened_sublist = self.flatten(item)
                flat_list.extend(flattened_sublist)
        return flat_list

    def calculate_bm25_scores(
        self, doc_id, term_freqs_token, term_freqs_token_title, idf_per_token
    ):
        doc_length = self.document_lengths[doc_id]
        title_length = self.title_lengths[doc_id]

        term_weights_body = [
            self.calculate_term_weight_body(
                tf=tf,
                doc_length=doc_length,
            )
            for tf in term_freqs_token
        ]

        term_weights_title = [
            self.calculate_term_weight_title(
                tf=tf,
                doc_length=title_length,
            )
            for tf in term_freqs_token_title
        ]

        weight_title = 2.0
        term_weights = [
            body_weight + title_weight * weight_title
            for body_weight, title_weight in zip(term_weights_body, term_weights_title)
        ]

        body_score = self.fielded_bm25_score(
            idf_tokens=idf_per_token, tf_tokens=term_weights_body
        )

        title_score = self.fielded_bm25_score(
            idf_tokens=idf_per_token, tf_tokens=term_weights_title
        )

        score = self.fielded_bm25_score(
            idf_tokens=idf_per_token,
            tf_tokens=term_weights,
        )

        return score, body_score, title_score

    def traditional_doc_ids_contain_semantic_doc_ids(
        self, traditional_doc_ids: list[int], semantic_doc_ids: list[int]
    ) -> list[SemTradContainResult]:
        semantic_doc_ids_sort = sorted(
            [
                (semantic_doc_id, idx)
                for idx, semantic_doc_id in enumerate(semantic_doc_ids)
            ],
            key=lambda x: x[0],
        )
        traditional_index = 0
        semantic_index = 0
        result: list[SemTradContainResult] = []

        while traditional_index < len(traditional_doc_ids) and semantic_index < len(
            semantic_doc_ids_sort
        ):
            if (
                traditional_doc_ids[traditional_index]
                == semantic_doc_ids_sort[semantic_index][0]
            ):
                result.append(
                    SemTradContainResult(
                        org_sem_idx=semantic_doc_ids_sort[semantic_index][1],
                        doc_id=traditional_doc_ids[traditional_index],
                        trad_idx=traditional_index,
                    )
                )
                traditional_index += 1
                semantic_index += 1
            elif (
                traditional_doc_ids[traditional_index]
                < semantic_doc_ids_sort[semantic_index][0]
            ):
                traditional_index += 1
            else:
                result.append(
                    SemTradContainResult(
                        org_sem_idx=semantic_doc_ids_sort[semantic_index][1],
                        doc_id=semantic_doc_ids_sort[semantic_index][0],
                        trad_idx=None,
                    )
                )
                semantic_index += 1

        return result

    def semantic_search(
        self, query: str, num_return: int, snippet_length: int
    ) -> tuple[int, list[tuple[float, DocumentInfo]]]:
        query_embedding = self.sentence_transformer.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # tokenize query
        # for each token get the doc list and term frequencies and postion offsets
        # intersection between doc_ids of semantic search and each token if semantic search doc id not in tradtional then 0 (save indices for term frequencies and position offsets)
        # score = some linear combination of rank and semantic score

        tokenized_query = tokenize_text(query)

        doc_ids_per_token: list[tuple[int, ...]] = []
        pos_offset_list_per_token: list[tuple[int, ...]] = []
        term_frequencies_per_token: list[tuple[int, ...]] = []
        term_frequencies_title_per_token: list[tuple[int, ...]] = []

        doc_freqs: list[int] = []

        for token in tokenized_query:
            doc_list, pos_offset_list, term_frequencies, term_frequencies_title = (
                self.get_docs(token, enable_threshold=False)
            )
            doc_ids_per_token.append(doc_list)
            pos_offset_list_per_token.append(pos_offset_list)
            term_frequencies_per_token.append(term_frequencies)
            term_frequencies_title_per_token.append(term_frequencies_title)

            doc_freqs.append(len(doc_list))

        cosine_similarities = util.semantic_search(
            query_embedding,  # pyright: ignore
            self.doc_embeddings,  # pyright: ignore
            top_k=num_return,
        )

        doc_ids = [int(match["corpus_id"]) for match in cosine_similarities[0]]

        matched_doc_ids: list[int] = doc_ids
        matched_term_freqs: list[list[int]] = [[] for _ in range(len(doc_ids))]
        matched_term_freqs_title: list[list[int]] = [[] for _ in range(len(doc_ids))]
        matched_pos_offsets: list[list[int | None]] = [[] for _ in range(len(doc_ids))]

        matched_bm25_scores: list[float] = []
        matched_bm25_scores_body: list[float] = []
        matched_bm25_scores_title: list[float] = []

        idf_per_token = [
            self.calculate_idf(self.metadata["num_docs"], df) for df in doc_freqs
        ]

        # matched_term_freqs = [doc1: [tf_token1, tf_token2]. doc2: [tf_token1, tf_token2]]]
        # matched_term_freq = [[], [],] if we have 2 doc ids
        for token_idx, token_doc_ids in enumerate(doc_ids_per_token):
            doc_ids_filtered = self.traditional_doc_ids_contain_semantic_doc_ids(
                list(token_doc_ids), doc_ids
            )

            for org_sem_idx, doc_id, traditional_index in doc_ids_filtered:
                if traditional_index is not None:
                    pos_offset = pos_offset_list_per_token[token_idx][traditional_index]
                    term_freq = term_frequencies_per_token[token_idx][traditional_index]
                    term_freq_title = term_frequencies_title_per_token[token_idx][
                        traditional_index
                    ]
                else:
                    pos_offset = None
                    term_freq = 0
                    term_freq_title = 0

                matched_term_freqs[org_sem_idx].append(term_freq)
                matched_term_freqs_title[org_sem_idx].append(term_freq_title)
                matched_pos_offsets[org_sem_idx].append(pos_offset)

        for i in range(len(matched_doc_ids)):
            score, body_score, title_score = self.calculate_bm25_scores(
                doc_id=matched_doc_ids[i],
                term_freqs_token=matched_term_freqs[i],
                term_freqs_token_title=matched_term_freqs_title[i],
                idf_per_token=idf_per_token,
            )
            matched_bm25_scores.append(score)
            matched_bm25_scores_body.append(body_score)
            matched_bm25_scores_title.append(title_score)

        scores = np.array([float(match["score"]) for match in cosine_similarities[0]])
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        features = self.extract_features(
            matched_doc_ids=matched_doc_ids,
            matched_bm25_scores=matched_bm25_scores,
            matched_bm25_scores_body=matched_bm25_scores_body,
            matched_bm25_scores_title=matched_bm25_scores_title,
            matched_term_freqs=matched_term_freqs,
            matched_term_freqs_title=matched_term_freqs_title,
            matched_pos_offsets=matched_pos_offsets,
        ).unsqueeze(0)

        with torch.no_grad():
            predicted_scores = self.ranking_model(features).squeeze(0).detach().numpy()

        predicted_scores = (predicted_scores - predicted_scores.min()) / (
            predicted_scores.max() - predicted_scores.min() + 1e-8
        )

        new_scores = scores + (np.pow(np.e, predicted_scores) - 1)
        doc_ids_sorted = sorted(
            list(zip(new_scores, matched_doc_ids)), key=lambda x: x[0], reverse=True
        )

        doc_infos = [
            (doc_id[0], self.get_doc_info(doc_id[1], snippet_length)) for doc_id in doc_ids_sorted
        ]

        return len(doc_ids_sorted), list(doc_infos)

    def traditional_search(
        self,
        query: str,
        mode: SearchMode,
        num_bm25_candidates: int,
        num_return: int,
        snippet_length: int,
    ):
        tokens = tokenize_text(query)

        doc_list: list[tuple[int, ...]] = []
        pos_offset_list: list[tuple[int, ...]] = []
        term_freqs: list[tuple[int, ...]] = []
        term_freqs_title: list[tuple[int, ...]] = []
        doc_freqs: list[int] = []
        match mode:
            case SearchMode.PHRASE:
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
                        pos_offset_list_per_token,
                        term_freq_per_token,
                        term_freq_title_per_token,
                    ) = self.get_docs(token)
                    if len(doc_list_per_token) > 0:
                        doc_list.append(doc_list_per_token)
                        pos_offset_list.append(pos_offset_list_per_token)
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
            (
                matched_doc_ids,
                matched_pos_offsets,
                matched_term_freqs,
                matched_term_freqs_title,
            ) = self.and_statement(
                doc_list, pos_offset_list, term_freqs, term_freqs_title
            )
        elif mode == SearchMode.OR:
            (
                matched_doc_ids,
                matched_pos_offsets,
                matched_term_freqs,
                matched_term_freqs_title,
            ) = self.or_statement(
                doc_list, pos_offset_list, term_freqs, term_freqs_title
            )
        elif mode == SearchMode.NOT:
            (
                matched_doc_ids,
                matched_pos_offsets,
                matched_term_freqs,
                matched_term_freqs_title,
            ) = self.not_statement(doc_list)
        elif mode == SearchMode.PHRASE:
            (
                matched_doc_ids,
                matched_pos_offsets,
                matched_term_freqs,
                matched_term_freqs_title,
            ) = self.phrase_statement(
                doc_list, pos_offset_list, term_freqs, term_freqs_title
            )
        elif mode == SearchMode.QUERY_EVALUATOR:
            (
                doc_freqs,
                matched_doc_ids,
                matched_pos_offsets,
                matched_term_freqs,
                matched_term_freqs_title,
            ) = self.query_evaluator(tokens)

        bm25_candidates: list[
            tuple[
                float,
                int,
                float,
                float,
                Sequence[int],
                Sequence[int],
                Sequence[int],
            ]
        ] = []

        if len(matched_term_freqs) == 1 and len(matched_doc_ids) != 1:
            matched_term_freqs = list(zip(*matched_term_freqs))

        assert len(matched_doc_ids) == len(matched_term_freqs)

        idf_per_token = [
            self.calculate_idf(self.metadata["num_docs"], df) for df in doc_freqs
        ]

        bm25_candidates = []

        for doc_id, term_freqs_token, term_freqs_token_title, pos_offsets in zip(
            matched_doc_ids,
            matched_term_freqs,
            matched_term_freqs_title,
            matched_pos_offsets,
        ):
            term_freqs_token = self.flatten(term_freqs_token)
            term_freqs_token_title = self.flatten(term_freqs_token_title)

            score, body_score, title_score = self.calculate_bm25_scores(
                doc_id=doc_id,
                term_freqs_token=term_freqs_token,
                term_freqs_token_title=term_freqs_token_title,
                idf_per_token=idf_per_token,
            )

            if len(bm25_candidates) < num_bm25_candidates:
                heapq.heappush(
                    bm25_candidates,
                    (
                        score,
                        doc_id,
                        body_score,
                        title_score,
                        term_freqs_token,
                        term_freqs_token_title,
                        pos_offsets,
                    ),
                )
            elif score > bm25_candidates[0][0]:
                heapq.heapreplace(
                    bm25_candidates,
                    (
                        score,
                        doc_id,
                        body_score,
                        title_score,
                        term_freqs_token,
                        term_freqs_token_title,
                        pos_offsets,
                    ),
                )

        results: list[tuple[float, DocumentInfo]] = []
        bm25_candidates = sorted(bm25_candidates, key=lambda x: x[0], reverse=True)

        bm25_candidates_doc_ids = []
        bm25_candidates_scores = []
        bm25_candidates_scores_body = []
        bm25_candidates_scores_title = []
        bm25_candidates_term_freqs = []
        bm25_candidates_term_freqs_title = []
        bm25_candidates_pos_offsets = []

        for (
            score,
            doc_id,
            body_score,
            title_score,
            term_freqs_token,
            term_freqs_token_title,
            pos_offsets,
        ) in bm25_candidates:
            bm25_candidates_doc_ids.append(doc_id)
            bm25_candidates_scores.append(score)
            bm25_candidates_scores_body.append(body_score)
            bm25_candidates_scores_title.append(title_score)
            bm25_candidates_term_freqs.append(term_freqs_token)
            bm25_candidates_term_freqs_title.append(term_freqs_token_title)
            bm25_candidates_pos_offsets.append(pos_offsets)

        result_candidates = []
        features = self.extract_features(
            matched_doc_ids=bm25_candidates_doc_ids,
            matched_bm25_scores=bm25_candidates_scores,
            matched_bm25_scores_body=bm25_candidates_scores_body,
            matched_bm25_scores_title=bm25_candidates_scores_title,
            matched_term_freqs=bm25_candidates_term_freqs,
            matched_term_freqs_title=bm25_candidates_term_freqs_title,
            matched_pos_offsets=bm25_candidates_pos_offsets,
        ).unsqueeze(0)

        with torch.no_grad():
            predicted_scores = self.ranking_model(features).squeeze(0).tolist()

        for doc_id, predicted_score in zip(bm25_candidates_doc_ids, predicted_scores):
            if len(result_candidates) < num_return:
                heapq.heappush(
                    result_candidates,
                    (predicted_score, doc_id),
                )
            elif predicted_score > result_candidates[0][0]:
                heapq.heapreplace(
                    result_candidates,
                    (predicted_score, doc_id),
                )

        result_candidates = sorted(result_candidates, key=lambda x: x[0], reverse=True)
        for score, doc_id in result_candidates:
            results.append((score, self.get_doc_info(doc_id, snippet_length)))

        return len(matched_doc_ids), results

    def search(
        self,
        query: str,
        mode: SearchMode,
        num_bm25_candidates: int = 100,
        num_return: int = 10,
        snippet_length: int = 100,
    ) -> tuple[int, list[tuple[float, DocumentInfo]]]:
        if self.enable_semantic_search:
            self.enable_spelling_correction = False
            return self.semantic_search(query, num_return, snippet_length)
        else:
            return self.traditional_search(
                query=query,
                mode=mode,
                num_bm25_candidates=num_bm25_candidates,
                num_return=num_return,
                snippet_length=snippet_length,
            )
