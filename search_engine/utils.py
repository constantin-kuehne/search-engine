from enum import Enum
from typing import NamedTuple


class SearchMode(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"
    QUERY_EVALUATOR = "QUERY_EVALUATOR"


POSTING = tuple[list[int], list[list[int]]]


class DocumentInfo(NamedTuple):
    original_docid: str
    url: str
    title: str


class SearchResult(NamedTuple):
    doc_id: int
    original_docid: str
    url: str
    title: str
