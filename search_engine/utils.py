import struct
from enum import Enum
from typing import NamedTuple

LAST_UTF8_CODE_POINT = "ÿ"
LAST_UNICODE_CODE_POINT = "\U0010FFFF"

INT_SIZE = 4
LONG_SIZE = 8


def get_length_from_bytes(bytes_array, offset: int) -> int:
    return struct.unpack("I", bytes_array[offset : offset + INT_SIZE])[0]


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
