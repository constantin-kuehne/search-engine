import struct
from enum import Enum
from typing import NamedTuple

LAST_UTF8_CODE_POINT = "ÿ"
LAST_UNICODE_CODE_POINT = "\U0010ffff"

INT_SIZE = 4
LONG_SIZE = 8


def get_length_from_bytes(bytes_array, offset: int) -> int:
    return struct.unpack("I", bytes_array[offset : offset + INT_SIZE])[0]

def get_trigrams_from_token(
    token: str
) -> set[str]:
    trigrams: set[str] = set()
    beginning_pos: int = 0
    ending_pos: int = 1
    token_size: int = len(token)

    if token_size == 1:
        trigrams.add("$" + token + "$")
        return trigrams

    while beginning_pos < token_size:
        if ending_pos == token_size:
            trigrams.add(token[beginning_pos : ending_pos] + "$")
            return trigrams

        trigram: str = ""
        if ending_pos == 1:
            trigram += "$"

        trigram += token[beginning_pos : ending_pos + 1]
        trigrams.add(trigram)

        ending_pos += 1
        if ending_pos > 2:
            beginning_pos += 1

    return trigrams

class SearchMode(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"
    QUERY_EVALUATOR = "QUERY_EVALUATOR"
    SEMANTIC = "SEMANTIC"

    def __repr__(self) -> str:
        return self.value


POSTING = tuple[list[int], list[list[int]], list[list[int]]]


class DocumentInfo(NamedTuple):
    original_docid: str
    url: str
    title: str
    body_snippet: str
