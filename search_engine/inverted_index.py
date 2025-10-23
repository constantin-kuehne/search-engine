from dataclasses import dataclass
from typing import TypedDict

class Positions(TypedDict):
    doc_id: int
    positions: list[int]

@dataclass
class InvertedIndex:
    index: dict[str, tuple[int, Positions]]  # term -> (document frequency, {doc_id: [positions]})

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        for position, term in enumerate(tokens):
            if term not in self.index:
                self.index[term] = (1, Positions(doc_id=doc_id, positions=[position]))
            else:
                doc_freq, position_list = self.index[term]


