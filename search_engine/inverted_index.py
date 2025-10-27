from search_engine.preprocessing import tokenize_text
from typing import NamedTuple

POSITIONS = dict[int, list[int]]


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

    def search(
        self, query: str, num_return: int = 10
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

        if len(doc_list) == 0:
            matched = []
        elif len(doc_list) == 1:
            matched = list(doc_list[0])
        else:
            matched = list(doc_list[0].intersection(*doc_list[1:]))

        results = [
            SearchResult(
                doc_id=doc_id, url=self.docs[doc_id].url, title=self.docs[doc_id].title
            )
            for doc_id in matched
        ]

        return len(matched), results[:num_return]
