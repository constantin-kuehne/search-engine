import bisect
import csv
import mmap
from pathlib import Path
import pickle
import struct
from operator import itemgetter
from typing import NamedTuple, Optional

from ordered_set import OrderedSet

from search_engine.preprocessing import (build_query_tree, shunting_yard,
                                         tokenize_text)
from search_engine.utils import (INT_SIZE, LONG_SIZE, POSTING, DocumentInfo, SearchMode,
                                 SearchResult, get_length_from_bytes)
import numpy as np


class InvertedIndex:
    def __init__(
        self,
        file_path_doc_id: str | Path,
        file_path_position_list: str | Path,
        file_path_term_index: str | Path,
        file_path_corpus_offset: str | Path,
        file_path_corpus: str | Path,
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
        # self.mm_postion_index = mmap.mmap(
        #     position_index_file.fileno(), length=0, prot=mmap.PROT_READ
        # )

        self.mm_corpus = mmap.mmap(corpus_file.fileno(), length=0, prot=mmap.PROT_READ)

        corpus_offset_file = open(file_path_corpus_offset, "rb")
        self.docs: dict[int, int] = pickle.load(corpus_offset_file)
        corpus_offset_file.close()

        doc_id_file.close()

        self.reader = lambda x: csv.DictReader(
            x, delimiter="\t", fieldnames=["docid", "url", "title", "body"]
        )

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

    def and_statement(self, doc_list: list[OrderedSet[int]]) -> list[int]:
        matched = []
        if len(doc_list) == 1:
            matched = list(doc_list[0])
        elif len(doc_list) > 1:
            matched = list(doc_list[0].intersection(*doc_list[1:]))

        return matched

    def or_statement(self, doc_list: list[OrderedSet[int]]) -> list[int]:
        matched = []
        if len(doc_list) == 1:
            matched = list(doc_list[0])
        elif len(doc_list) > 1:
            matched = list(doc_list[0].union(*doc_list[1:]))

        return matched

    def not_statement(self, doc_list: list[OrderedSet[int]]) -> list[int]:
        matched = []
        if len(doc_list) == 0:
            matched = list(self.docs.keys())
        else:
            matched = list(OrderedSet(self.docs.keys()).difference(*doc_list))

        return matched

    def intersection_phrase(
        self, docs_per_token: list[OrderedSet[int]], doc_pos_per_token: list[tuple[int]]
    ) -> tuple[OrderedSet[int], list[tuple[int]]]:
        intersection = docs_per_token[0].intersection(*docs_per_token[1:])
        indices_per_token = [
            [bisect.bisect_left(x, y) for y in intersection] for x in docs_per_token
        ]

        assert all(len(lst) == len(indices_per_token[0]) for lst in indices_per_token)

        new_pos_list_per_token: list[tuple[int]] = []
        for i, pos_list in enumerate(doc_pos_per_token):
            getter = itemgetter(*indices_per_token[i])
            new_pos_list: tuple[int] = getter(pos_list)
            if isinstance(new_pos_list, int):
                new_pos_list = (new_pos_list,)
            new_pos_list_per_token.append(new_pos_list)


        print(len(new_pos_list_per_token[0]))
        assert all(len(lst) == len(new_pos_list_per_token[0]) for lst in new_pos_list_per_token)

        new_pos_list_per_doc = np.array(new_pos_list_per_token).T.tolist()
        return intersection, new_pos_list_per_doc

    def phrase_statement(
        self,
        docs_per_token: list[OrderedSet[int]],
        doc_pos_offset_per_token: list[tuple[int]],
    ) -> list[int]:
        matched = []
        if len(docs_per_token) == 1:
            matched = list(docs_per_token[0])
        elif len(docs_per_token) > 1:
            match_candidates, pos_tokens_per_doc_candidate = self.intersection_phrase(
                docs_per_token, doc_pos_offset_per_token
            )

            pos_list_tokens_per_doc: list[list[tuple[int]]] = []
            for pos_offset_tuple in pos_tokens_per_doc_candidate:
                pos_list_token: list[tuple[int]] = []
                for pos_offset in pos_offset_tuple:
                    length_pos_list = struct.unpack(
                        "I", self.mm_position_list[pos_offset: pos_offset + INT_SIZE]
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

            matched = []
            for doc_id, pos_list_per_token in zip(
                match_candidates, pos_list_tokens_per_doc
            ):
                if self.has_phrase(pos_list_per_token):
                    matched.append(doc_id)

        return matched

    def evaluate_subtree(self, node) -> OrderedSet[int]:
        if isinstance(node.value, SearchMode):
            if node.value == SearchMode.AND:
                left_result = self.evaluate_subtree(node.left)
                right_result = self.evaluate_subtree(node.right)
                return OrderedSet(self.and_statement([left_result, right_result]))
            elif node.value == SearchMode.OR:
                left_result = self.evaluate_subtree(node.left)
                right_result = self.evaluate_subtree(node.right)
                return OrderedSet(self.or_statement([left_result, right_result]))
            elif node.value == SearchMode.NOT:
                left_result = self.evaluate_subtree(node.left)
                return OrderedSet(self.not_statement([left_result]))

        if isinstance(node.value, list):
            # phrase search
            tokens = node.value
            doc_list: list[OrderedSet[int]] = []
            pos_offset_list: list[tuple[int]] = []
            for token in tokens:
                docs = self.get_docs_phrase(token)
                doc_list.append(docs[0])
                pos_offset_list.append(docs[1])
            return OrderedSet(self.phrase_statement(doc_list, pos_offset_list))

        if isinstance(node.value, str):
            return self.get_docs(node.value)

        return OrderedSet([])

    def query_evaluator(self, tokens: list[str]) -> list[int]:
        matched: list[int] = []

        output_queue = shunting_yard(tokens)
        root = build_query_tree(output_queue)
        matched = list(self.evaluate_subtree(root))

        return matched

    def get_docs(
        self,
        token: str,
    ) -> OrderedSet[int]:
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
            return OrderedSet(doc_list)
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            return OrderedSet([])

    def get_docs_phrase(
        self,
        token: str,
    ) -> tuple[OrderedSet[int], tuple[int]]:
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
            pos_offset_list: tuple[int] = struct.unpack(
                f"{length_doc_list}Q",
                self.mm_doc_id_list[
                    res + INT_SIZE + length_doc_list * INT_SIZE : res
                    + INT_SIZE
                    + length_doc_list * INT_SIZE
                    + length_doc_list * LONG_SIZE
                ],
            )

            return OrderedSet(doc_list), pos_offset_list
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            empty_tuple: tuple[int] = tuple([])
            return OrderedSet([]), empty_tuple

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

    def search(
        self, query: str, mode: SearchMode, num_return: int = 10
    ) -> tuple[int, list[SearchResult]]:
        tokens = tokenize_text(query)

        match mode:
            case SearchMode.PHRASE:
                doc_list: list[OrderedSet[int]] = []
                pos_offset_list: list[tuple[int]] = []
                for token in tokens:
                    docs = self.get_docs_phrase(token)
                    doc_list.append(docs[0])
                    pos_offset_list.append(docs[1])
            case SearchMode.AND | SearchMode.OR | SearchMode.NOT:
                doc_list: list[OrderedSet[int]] = []
                for token in tokens:
                    doc_list.append(self.get_docs(token))
            case SearchMode.QUERY_EVALUATOR:
                pass
            case _:
                raise ValueError(f"Unsupported search mode: {mode}")

        matched = []
        if mode == SearchMode.AND:
            matched = self.and_statement(doc_list)
        elif mode == SearchMode.OR:
            matched = self.or_statement(doc_list)
        elif mode == SearchMode.NOT:
            matched = self.not_statement(doc_list)
        elif mode == SearchMode.PHRASE:
            matched = self.phrase_statement(doc_list, pos_offset_list)
        elif mode == SearchMode.QUERY_EVALUATOR:
            matched = self.query_evaluator(tokens)

        results = []
        for doc_id in matched[:num_return]:
            doc_info = self.get_doc_info(doc_id)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    original_docid=doc_info.original_docid,
                    url=doc_info.url,
                    title=doc_info.title,
                )
            )

        return len(matched), results
