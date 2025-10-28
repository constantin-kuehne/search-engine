from search_engine.preprocessing import tokenize_text
from typing import NamedTuple
from enum import Enum

POSITIONS = dict[int, list[int]]


class DocumentInfo(NamedTuple):
    url: str
    title: str


class SearchResult(NamedTuple):
    doc_id: int
    url: str
    title: str


class SearchMode(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[
            str, tuple[int, POSITIONS]
        ] = {}  # term -> (document frequency, {doc_id: [positions]})
        self.docs: dict[int, DocumentInfo] = {}

    def add_document(
        self, doc_id: int, url: str, title: str, tokens: list[str]
    ) -> None:
        self.docs[doc_id] = DocumentInfo(url=url, title=title)
        for position, term in enumerate(tokens):
            if term not in self.index:
                self.index[term] = (1, {doc_id: [position]})
            else:
                doc_freq, position_list = self.index[term]
                if doc_id not in position_list.keys():
                    doc_freq += 1
                    position_list[doc_id] = [position]
                else:
                    position_list[doc_id].append(position)

                self.index[term] = (doc_freq, position_list)

    def has_phrase(self, doc_id: int, tokens: list[str]) -> bool:
        pos_lists = []
        for token in tokens:
            pos_lists.append(self.index[token][1][doc_id])

        indices = [0 for _ in range(len(pos_lists))]
        print(indices)
        has_phrase: bool = False

        for _ in range(len(pos_lists[0])):
            for i, pos_list in enumerate(pos_lists[1:]):
                print(indices[i + 1])
                print(f"{len(pos_list)=}")
                while (
                    pos_list[indices[i + 1]] < pos_lists[i][indices[i]]
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

    def search(
        self, query: str, mode: SearchMode, num_return: int = 10
    ) -> tuple[int, list[SearchResult]]:
        tokens = tokenize_text(query)

        doc_list: list[set[int]] = []
        for token in tokens:
            res = self.index.get(token, None)
            if res is not None:
                doc_freq, position_dict = res
                doc_list.append(set(position_dict.keys()))
            else:
                # add the empty set if term not found, so we give no results
                # the correct AND semantic
                doc_list.append(set())

        matched = []
        if mode == SearchMode.AND:
            if len(doc_list) == 1:
                matched = list(doc_list[0])
            elif len(doc_list) > 1:
                matched = list(doc_list[0].intersection(*doc_list[1:]))
        elif mode == SearchMode.OR:
            if len(doc_list) == 1:
                matched = list(doc_list[0])
            elif len(doc_list) > 1:
                matched = list(doc_list[0].union(*doc_list[1:]))
        elif mode == SearchMode.NOT:
            if len(doc_list) == 0:
                matched = list(self.docs.keys())
            else:
                matched = list(set(self.docs.keys()).difference(*doc_list))
        elif mode == SearchMode.PHRASE:
            if len(doc_list) == 1:
                print(doc_list)
                matched = list(doc_list[0])
            elif len(doc_list) > 1:
                match_candidates = doc_list[0].intersection(*doc_list[1:])
                matched = []
                for doc_id in match_candidates:
                    if self.has_phrase(doc_id, tokens):
                        matched.append(doc_id)

        results = [
            SearchResult(
                doc_id=doc_id, url=self.docs[doc_id].url, title=self.docs[doc_id].title
            )
            for doc_id in matched
        ]

        return len(matched), results[:num_return]
