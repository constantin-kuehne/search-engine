import csv
import heapq
import math
import mmap
import pickle
import struct
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

from ordered_set import OrderedSet

from search_engine.preprocessing import (build_query_tree, shunting_yard,
                                         tokenize_text)
from search_engine.utils import (INT_SIZE, LONG_SIZE, DocumentInfo, SearchMode,
                                 SearchResult, get_length_from_bytes)


class InvertedIndex:
    def __init__(
        self,
        file_path_doc_id: str | Path,
        file_path_position_list: str | Path,
        file_path_term_index: str | Path,
        file_path_corpus_offset: str | Path,
        file_path_corpus: str | Path,
        file_path_metadata: str | Path,
        file_path_document_lengths: str | Path,
    ) -> None:
        doc_id_file = open(file_path_doc_id, "rb")
        term_index_file = open(file_path_term_index, "rb")
        position_list_file = open(file_path_position_list, mode="rb")
        # position_index_file = open(file_path_position_index, mode="rb")

        corpus_file = open(file_path_corpus, "rb")

        self.index_2 = pickle.load(term_index_file)  # TODO: Set to self.index

        self.mm_doc_id_list = mmap.mmap(
            doc_id_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        self.mm_position_list = mmap.mmap(
            position_list_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        self.mm_corpus = mmap.mmap(corpus_file.fileno(), length=0, prot=mmap.PROT_READ)

        corpus_offset_file = open(file_path_corpus_offset, "rb")
        self.docs: dict[int, int] = pickle.load(corpus_offset_file)

        corpus_offset_file.close()

        term_index_file.close()

        self.reader = lambda x: csv.DictReader(
            x, delimiter="\t", fieldnames=["docid", "url", "title", "body"]
        )

        with open(file_path_metadata, "rb") as f:
            self.metadata = pickle.load(f)

        with open(file_path_document_lengths, "rb") as f:
            self.document_lengths = pickle.load(f)

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
    ) -> tuple[list[int], list[list[int]]]:
        doc_ids = [doc_list for doc_list in doc_ids if len(doc_list) > 0]
        pointer = [0 for _ in range(len(doc_ids))]
        result_doc_ids = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids):
            heapq.heappush(min_heap, (doc_list[0], i, term_frequencies[i][0]))

        counter_same_value = 0
        last_min = -1
        result_term_freqs: list[list[int]] = []
        last_term_freqs: list[int] = []

        while min_heap:
            current_min, current_index, term_frequency = heapq.heappop(min_heap)

            if last_min == current_min:
                counter_same_value += 1
            else:
                last_term_freqs = []
                counter_same_value = 0

            last_term_freqs.append(term_frequency)
            if counter_same_value == len(doc_ids) - 1:
                result_term_freqs.append(last_term_freqs)
                result_doc_ids.append(current_min)

            pointer[current_index] += 1
            if len(doc_ids[current_index]) <= pointer[current_index]:
                last_min = current_min
                continue

            heapq.heappush(
                min_heap,
                (
                    doc_ids[current_index][pointer[current_index]],
                    current_index,
                    term_frequencies[current_index][pointer[current_index]],
                ),
            )
            last_min = current_min
        return result_doc_ids, result_term_freqs

    def intersection_phrase(
        self,
        doc_ids_per_token: list[tuple[int, ...]],
        doc_pos_per_token: list[tuple[int, ...]],
        term_frequencies: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]], Sequence[Sequence[int]]]:
        pointer = [0 for _ in range(len(doc_ids_per_token))]
        result_doc_ids: list[int] = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids_per_token):
            heapq.heappush(
                min_heap,
                (doc_list[0], i, term_frequencies[i][0], doc_pos_per_token[i][0]),
            )

        counter_same_value = 0
        last_min = -1

        result_term_freqs: list[list[int]] = []
        last_term_freqs: list[int] = []

        result_doc_pos_offsets: list[list[int]] = []
        last_doc_pos_offsets: list[int] = []

        while min_heap:
            current_min, current_index, term_frequency, doc_pos_offset = heapq.heappop(
                min_heap
            )

            if last_min == current_min:
                counter_same_value += 1
            else:
                last_term_freqs = []
                last_doc_pos_offsets = []
                counter_same_value = 0

            last_term_freqs.append(term_frequency)
            last_doc_pos_offsets.append(doc_pos_offset)

            if counter_same_value == len(doc_ids_per_token) - 1:
                result_term_freqs.append(last_term_freqs)
                result_doc_pos_offsets.append(last_doc_pos_offsets)
                result_doc_ids.append(current_min)

            pointer[current_index] += 1
            if len(doc_ids_per_token[current_index]) <= pointer[current_index]:
                last_min = current_min
                continue

            heapq.heappush(
                min_heap,
                (
                    doc_ids_per_token[current_index][pointer[current_index]],
                    current_index,
                    term_frequencies[current_index][pointer[current_index]],
                    doc_pos_per_token[current_index][pointer[current_index]],
                ),
            )
            last_min = current_min

        return result_doc_ids, result_term_freqs, result_doc_pos_offsets

    def union(
        self,
        doc_ids: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]]]:
        num_terms = len(doc_ids)
        pointers = [0] * num_terms
        result_doc_ids = []
        result_term_freqs = []
        min_heap = []

        for i, doc_list in enumerate(doc_ids):
            if doc_list:
                heapq.heappush(min_heap, (doc_list[0], i))

        last_doc_id = -1
        current_tf_vector = []
        while min_heap:
            doc_id, term_index = heapq.heappop(min_heap)
            if doc_id != last_doc_id:
                if last_doc_id != -1:
                    result_term_freqs.append(current_tf_vector)

                last_doc_id = doc_id
                result_doc_ids.append(doc_id)
                current_tf_vector = [0] * num_terms

            tf = term_frequencies[term_index][pointers[term_index]]
            current_tf_vector[term_index] = tf
            pointers[term_index] += 1

            if pointers[term_index] < len(doc_ids[term_index]):
                next_doc_id = doc_ids[term_index][pointers[term_index]]
                heapq.heappush(min_heap, (next_doc_id, term_index))

        if last_doc_id != -1:
            result_term_freqs.append(current_tf_vector)
        return result_doc_ids, result_term_freqs

    def and_statement(
        self,
        doc_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
    ) -> tuple[Sequence[int], Sequence[Sequence[int]]]:
        matched: tuple[Sequence[int], Sequence[Sequence[int]]] = tuple([])
        if len(doc_list) == 1:
            matched = (doc_list[0], list(zip(*term_frequencies)))
        elif len(doc_list) > 1:
            matched = self.intersection(doc_list, term_frequencies)

        return matched

    def or_statement(
        self,
        doc_list: Sequence[Sequence[int]],
        term_frequencies: Sequence[Sequence[int]],
    ) -> tuple[list[int], list[list[int]]]:
        matched: tuple[list[int], list[list[int]]] = tuple([])
        if len(doc_list) == 1:
            matched = (list(doc_list[0]), [list(term_frequencies[0])])
        elif len(doc_list) > 1:
            matched = self.union(doc_list, term_frequencies)

        return matched

    def not_statement(
        self, doc_list: Sequence[Sequence[int]], *args
    ) -> tuple[list[int], list[list[int]]]:
        matched: tuple[list[int], list[list[int]]] = tuple([])
        if len(doc_list) == 0:
            matched = (
                list(self.docs.keys()),
                [[0 for _ in range(len(self.docs.keys()))]],
            )
        else:
            doc_ids_matched = list(OrderedSet(self.docs.keys()).difference(*doc_list))
            matched = (doc_ids_matched, [[0 for _ in range(len(doc_ids_matched))]])

        return matched

    def phrase_statement(
        self,
        docs_per_token: list[tuple[int, ...]],
        doc_pos_offset_per_token: list[tuple[int]],
        term_freqs: Sequence[Sequence[int]],
    ) -> tuple[Sequence[int], Sequence[Sequence[int]]]:
        if len(docs_per_token) == 1:
            return docs_per_token[0], term_freqs

        matched: list[int] = []
        match_candidates, term_freqs_per_doc, pos_tokens_per_doc_candidate = (
            self.intersection_phrase(
                docs_per_token, doc_pos_offset_per_token, term_freqs
            )
        )

        if len(match_candidates) == 0:
            return matched, term_freqs_per_doc

        pos_list_tokens_per_doc: list[list[tuple[int]]] = []
        for pos_offset_tuple in pos_tokens_per_doc_candidate:
            pos_list_token: list[tuple[int]] = []
            for pos_offset in pos_offset_tuple:
                length_pos_list = struct.unpack(
                    "I", self.mm_position_list[pos_offset : pos_offset + INT_SIZE]
                )[0]
                pos_list: tuple[int] = struct.unpack(
                    f"{length_pos_list}I",
                    self.mm_position_list[
                        pos_offset + INT_SIZE : pos_offset
                        + INT_SIZE
                        + length_pos_list * INT_SIZE
                    ],
                )
                pos_list_token.append(pos_list)
            pos_list_tokens_per_doc.append(pos_list_token)

        term_freqs_per_doc_matched = []
        for doc_id, pos_list_per_token, term_freqs_doc in zip(
            match_candidates, pos_list_tokens_per_doc, term_freqs_per_doc
        ):
            if self.has_phrase(
                pos_list_per_token,
            ):
                matched.append(doc_id)
                term_freqs_per_doc_matched.append(term_freqs_doc)

        return (matched, term_freqs_per_doc_matched)

    def evaluate_subtree(
        self, node
    ) -> tuple[list[int], Sequence[int], Sequence[Sequence[int]]]:
        if isinstance(node.value, SearchMode):
            if node.value == SearchMode.AND:
                left_result_doc_freq, left_result_doc_list, left_result_term_freq = (
                    self.evaluate_subtree(node.left)
                )
                right_result_doc_freq, right_result_doc_list, right_result_term_freq = (
                    self.evaluate_subtree(node.right)
                )
                result_term_freq = list(left_result_term_freq)
                if not isinstance(node.left.value, str) and not node.left.value == SearchMode.NOT:
                    result_term_freq = [result_term_freq]
                if not isinstance(node.right.value, str) and not node.right.value == SearchMode.NOT:
                    right_result_term_freq = [right_result_term_freq]
                result_term_freq.extend(right_result_term_freq) # pyright: ignore

                result_doc_freq = list(left_result_doc_freq)
                result_doc_freq.extend(right_result_doc_freq)
                return (
                    result_doc_freq,
                    *self.and_statement(
                        [left_result_doc_list, right_result_doc_list],
                        result_term_freq, # pyright: ignore
                    ),
                )
            elif node.value == SearchMode.OR:
                left_result_doc_freq, left_result_doc_list, left_result_term_freq = (
                    self.evaluate_subtree(node.left)
                )
                right_result_doc_freq, right_result_doc_list, right_result_term_freq = (
                    self.evaluate_subtree(node.right)
                )
                result_term_freq = list(left_result_term_freq)
                result_term_freq.extend(right_result_term_freq)

                result_doc_freq = list(left_result_doc_freq)
                result_doc_freq.extend(right_result_doc_freq)
                return (
                    result_doc_freq,
                    *self.or_statement(
                        [left_result_doc_list, right_result_doc_list], result_term_freq
                    ),
                )
            elif node.value == SearchMode.NOT:
                left_result_doc_freq, left_result_doc_list, result_term_freq = (
                    self.evaluate_subtree(node.left)
                )
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
            doc_freqs_phrase: list[int] = []
            for token in tokens:
                doc_list_per_token, pos_offset_list_per_token, term_freq_per_token = (
                    self.get_docs_phrase(token)
                )
                doc_list_phrase.append(doc_list_per_token)
                pos_offset_list_phrase.append(pos_offset_list_per_token)
                term_freqs_phrase.append(term_freq_per_token)
                doc_freqs_phrase.append(len(doc_list_per_token))
            return (
                doc_freqs_phrase,
                *self.phrase_statement(
                    doc_list_phrase, pos_offset_list_phrase, term_freqs_phrase
                ),
            )

        if isinstance(node.value, str):
            doc_list, term_frequencies = self.get_docs(node.value)
            return [len(doc_list)], doc_list, [list(term_frequencies)]

        return tuple([])

    def query_evaluator(
        self, tokens: list[str]
    ) -> tuple[list[int], Sequence[int], Sequence[Sequence[int]]]:
        output_queue = shunting_yard(tokens)
        root = build_query_tree(output_queue)
        doc_freqs, matched_doc_ids, matched_doc_freqs = self.evaluate_subtree(root)
        return doc_freqs, matched_doc_ids, matched_doc_freqs

    def get_docs(
        self,
        token: str,
        idf_threshold: float = 1.5,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        res: Optional[int] = self.index_2.get(token, None)
        if res is not None:
            length_term: int = get_length_from_bytes(self.mm_doc_id_list, res)
            res += INT_SIZE + length_term  # move to the document list
            length_doc_list: int = get_length_from_bytes(self.mm_doc_id_list, res)
            if self.calculate_idf(self.metadata["num_docs"], length_doc_list) < idf_threshold:
                empty_tuple: tuple[int] = tuple([])
                return empty_tuple, empty_tuple

            doc_list = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    res + INT_SIZE : res + INT_SIZE + length_doc_list * INT_SIZE
                ],  # + 4 and * 4 because we are on bytes level, but we use uint32 which is 4 bytes
            )
            term_frequencies = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE : res
                    + INT_SIZE
                    + length_doc_list * INT_SIZE * 2
                ],
            )
            return doc_list, term_frequencies
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            empty_tuple: tuple[int] = tuple([])
            return empty_tuple, empty_tuple
    
    def calculate_idf(self, N: int, doc_freq: int) -> float:
        return math.log((N - doc_freq + 0.5) / (doc_freq + 0.5))

    def get_docs_phrase(
        self,
        token: str,
    ) -> tuple[tuple[int], tuple[int], tuple[int]]:
        res: Optional[int] = self.index_2.get(token, None)
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
            term_frequencies = struct.unpack(
                f"{length_doc_list}I",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE : res
                    + INT_SIZE
                    + length_doc_list * INT_SIZE * 2
                ],
            )
            pos_offset_list: tuple[int] = struct.unpack(
                f"{length_doc_list}Q",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE * 2 : res
                    + INT_SIZE
                    + length_doc_list
                    * INT_SIZE
                    * 2  # move to position offset list: times 2 because we have to skip doc id list and term frequency list
                    + length_doc_list * LONG_SIZE
                ],
            )

            return doc_list, pos_offset_list, term_frequencies
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            empty_tuple: tuple[int] = tuple([])
            return empty_tuple, empty_tuple, empty_tuple

    def get_doc_info(self, doc_id: int) -> DocumentInfo:
        offset = self.docs[doc_id]
        next_offset = self.docs.get(doc_id + 1, self.mm_corpus.size())
        line = self.mm_corpus[offset:next_offset].decode("utf-8")
        row = next(self.reader([line]))
        return DocumentInfo(
            original_docid=row["docid"],
            url=row["url"],
            title=row["title"],
            body=row["body"],
        )

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
        self, query: str, mode: SearchMode, num_return: int = 10, length_body: int = 50
    ) -> tuple[int, list[tuple[float, SearchResult]]]:
        tokens = tokenize_text(query)

        doc_list: list[tuple[int, ...]] = []
        term_freqs: list[tuple[int, ...]] = []
        doc_freqs: list[int] = []
        match mode:
            case SearchMode.PHRASE:
                pos_offset_list: list[tuple[int]] = []
                for token in tokens:
                    (
                        doc_list_per_token,
                        pos_offset_list_per_token,
                        term_freq_per_token,
                    ) = self.get_docs_phrase(token)
                    doc_list.append(doc_list_per_token)
                    pos_offset_list.append(pos_offset_list_per_token)
                    term_freqs.append(term_freq_per_token)
                    doc_freqs.append(len(doc_list_per_token))
            case SearchMode.AND | SearchMode.OR | SearchMode.NOT:
                for token in tokens:
                    doc_list_per_token, term_freq_per_token = self.get_docs(token)
                    doc_list.append(doc_list_per_token)
                    term_freqs.append(term_freq_per_token)
                    doc_freqs.append(len(doc_list_per_token))
            case SearchMode.QUERY_EVALUATOR:
                pass
            case _:
                raise ValueError(f"Unsupported search mode: {mode}")

        matched_doc_ids: Sequence[int] = []
        matched_term_freqs: Sequence[Sequence[int]] = []
        if mode == SearchMode.AND:
            matched_doc_ids, matched_term_freqs = self.and_statement(
                doc_list, term_freqs
            )
        elif mode == SearchMode.OR:
            matched_doc_ids, matched_term_freqs = self.or_statement(
                doc_list, term_freqs
            )
        elif mode == SearchMode.NOT:
            matched_doc_ids, matched_term_freqs = self.not_statement(doc_list)
        elif mode == SearchMode.PHRASE:
            matched_doc_ids, matched_term_freqs = self.phrase_statement(
                doc_list, pos_offset_list, term_freqs
            )
        elif mode == SearchMode.QUERY_EVALUATOR:
            doc_freqs, matched_doc_ids, matched_term_freqs = self.query_evaluator(
                tokens
            )

        results: list[tuple[float, SearchResult]] = []

        bm25_score = partial(
            self.bm25_score,
            df_tokens=doc_freqs,
            N=self.metadata["num_docs"],
            avg_length=self.metadata["average_doc_length"],
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

        if len(matched_term_freqs) == 1 and len(matched_doc_ids) != 1:
            matched_term_freqs = list(zip(*matched_term_freqs))
        assert len(matched_doc_ids) == len(matched_term_freqs)
        for doc_id, term_freqs_token in zip(matched_doc_ids, matched_term_freqs):
            term_freqs_token = flatten(term_freqs_token)
            doc_info = self.get_doc_info(doc_id)

            doc_length = self.document_lengths[doc_id]
            score = bm25_score(tf_tokens=term_freqs_token, doc_length=doc_length)

            results.append(
                (
                    score,
                    SearchResult(
                        doc_id=doc_id,
                        original_docid=doc_info.original_docid,
                        url=doc_info.url,
                        title=doc_info.title,
                        body=doc_info.body[:length_body],
                    ),
                )
            )
        results = sorted(results, key=lambda x: x[0], reverse=True)[:num_return]

        return len(matched_doc_ids), results
