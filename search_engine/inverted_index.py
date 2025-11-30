import bisect
import csv
import mmap
import pickle
import struct
from typing import NamedTuple, Optional

from search_engine.preprocessing import (build_query_tree, shunting_yard,
                                         tokenize_text)
from search_engine.utils import (INT_SIZE, POSTING, DocumentInfo, SearchMode,
                                 SearchResult, get_length_from_bytes)


class InvertedIndex:
    def __init__(
        self,
        file_path_doc_id: str,
        file_path_position_list: str,
        file_path_position_index: str,
        file_path_term_index: str,
        file_path_corpus_offset: str,
        file_path_corpus: str,
    ) -> None:
        doc_id_file = open(file_path_doc_id, "rb")
        term_index_file = open(file_path_term_index, "rb")
        # position_list_file = open(file_path_position_list, mode="rb")
        # position_index_file = open(file_path_position_index, mode="rb")

        corpus_file = open(file_path_corpus, "rb")

        self.index_2 = pickle.load(term_index_file)  # TODO: Set to self.index

        self.mm_doc_id_list = mmap.mmap(
            doc_id_file.fileno(), length=0, prot=mmap.PROT_READ
        )

        # self.mm_postion_list = mmap.mmap(
        #     position_list_file.fileno(), length=0, prot=mmap.PROT_READ
        # )
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

    def has_phrase(self, doc_id: int, tokens: list[str]) -> bool:
        raise NotImplementedError("Phrase search not implemented yet.")
        pos_lists = []
        for token in tokens:
            idx = bisect.bisect_left(self.index[token][0], doc_id)
            pos_lists.append(self.index[token][1][idx])

        indices = [0 for _ in range(len(pos_lists))]
        has_phrase: bool = False

        for _ in range(len(pos_lists[0])):
            for i, pos_list in enumerate(pos_lists[1:]):
                while (
                    pos_list[indices[i + 1]] <= pos_lists[i][indices[i]]
                ):  # +1 because we skip first list
                    indices[i + 1] += 1

                    if indices[i + 1] >= len(pos_list):
                        return False

                if pos_list[indices[i + 1]] == pos_lists[i][indices[i]] + 1:
                    has_phrase = True
                else:
                    has_phrase = False
                    break

            if has_phrase:
                break

            indices[0] += 1

        return has_phrase

    def and_statement(self, doc_list: list[set[int]]) -> list[int]:
        matched = []
        if len(doc_list) == 1:
            matched = list(doc_list[0])
        elif len(doc_list) > 1:
            matched = list(doc_list[0].intersection(*doc_list[1:]))

        return matched

    def or_statement(self, doc_list: list[set[int]]) -> list[int]:
        matched = []
        if len(doc_list) == 1:
            matched = list(doc_list[0])
        elif len(doc_list) > 1:
            matched = list(doc_list[0].union(*doc_list[1:]))

        return matched

    def not_statement(self, doc_list: list[set[int]]) -> list[int]:
        matched = []
        if len(doc_list) == 0:
            matched = list(self.docs.keys())
        else:
            matched = list(set(self.docs.keys()).difference(*doc_list))

        return matched

    def phrase_statement(
        self, doc_list: list[set[int]], tokens: list[str]
    ) -> list[int]:
        matched = []
        if len(doc_list) == 1:
            matched = list(doc_list[0])
        elif len(doc_list) > 1:
            match_candidates = doc_list[0].intersection(*doc_list[1:])
            matched = []
            for doc_id in match_candidates:
                if self.has_phrase(doc_id, tokens):
                    matched.append(doc_id)

        return matched

    def evaluate_subtree(self, node) -> set[int]:
        if isinstance(node.value, SearchMode):
            if node.value == SearchMode.AND:
                left_result = self.evaluate_subtree(node.left)
                right_result = self.evaluate_subtree(node.right)
                return set(self.and_statement([left_result, right_result]))
            elif node.value == SearchMode.OR:
                left_result = self.evaluate_subtree(node.left)
                right_result = self.evaluate_subtree(node.right)
                return set(self.or_statement([left_result, right_result]))
            elif node.value == SearchMode.NOT:
                left_result = self.evaluate_subtree(node.left)
                return set(self.not_statement([left_result]))

        if isinstance(node.value, list):
            # phrase search
            tokens = node.value
            doc_list: list[set[int]] = []
            for token in tokens:
                doc_list.append(self.get_docs(token))
            return set(self.phrase_statement(doc_list, tokens))

        if isinstance(node.value, str):
            return self.get_docs(node.value)

        return set()

    def query_evaluator(self, tokens: list[str]) -> list[int]:
        matched: list[int] = []

        output_queue = shunting_yard(tokens)
        root = build_query_tree(output_queue)
        matched = list(self.evaluate_subtree(root))

        return matched

    def get_docs(self, token: str) -> set[int]:
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
            return set(doc_list)
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            return set()

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

        if mode != SearchMode.QUERY_EVALUATOR:
            doc_list: list[set[int]] = []
            for token in tokens:
                doc_list.append(self.get_docs(token))

        matched = []
        if mode == SearchMode.AND:
            matched = self.and_statement(doc_list)
        elif mode == SearchMode.OR:
            matched = self.or_statement(doc_list)
        elif mode == SearchMode.NOT:
            matched = self.not_statement(doc_list)
        elif mode == SearchMode.PHRASE:
            matched = self.phrase_statement(doc_list, tokens)
        elif mode == SearchMode.QUERY_EVALUATOR:
            print(tokens)
            matched = self.query_evaluator(tokens)

        results = []
        for doc_id in matched:
            doc_info = self.get_doc_info(doc_id)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    original_docid=doc_info.original_docid,
                    url=doc_info.url,
                    title=doc_info.title,
                )
            )

        return len(matched), results[:num_return]
