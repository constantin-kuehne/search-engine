from typing import NamedTuple

from search_engine.preprocessing import (build_query_tree, shunting_yard,
                                         tokenize_text)
from search_engine.utils import SearchMode

POSTING = tuple[list[int], list[list[int]]]


class DocumentInfo(NamedTuple):
    url: str
    title: str


class SearchResult(NamedTuple):
    doc_id: int
    url: str
    title: str


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[
            str, POSTING
        ] = {}  # term -> (document list, [postion list])
        # TODO: use list instead of dict for document ids

        # simplemma + woosh

        self.docs: dict[int, DocumentInfo] = {}

    def add_document(
        self, doc_id: int, url: str, title: str, tokens: list[str]
    ) -> None:
        self.docs[doc_id] = DocumentInfo(url=url, title=title)
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

    def has_phrase(self, doc_id: int, tokens: list[str]) -> bool:
        pos_lists = []
        for token in tokens:
            idx = self.index[token][0].index(doc_id)
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
        res = self.index.get(token, None)
        if res is not None:
            doc_list, position_list_list = res
            return set(doc_list)
        else:
            # add the empty set if term not found, so we give no results
            # the correct AND semantic
            return set()

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

        results = [
            SearchResult(
                doc_id=doc_id, url=self.docs[doc_id].url, title=self.docs[doc_id].title
            )
            for doc_id in matched
        ]

        return len(matched), results[:num_return]
