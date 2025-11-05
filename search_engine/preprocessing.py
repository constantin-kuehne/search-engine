from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import spacy

nlp = spacy.blank("en")


special_tokens = ["AND", "&", "OR", "|", '"', "(", ")"]


@dataclass
class QueryNode:
    value: str
    left: Optional[QueryNode] = None
    right: Optional[QueryNode] = None


def tokenize_text(text: str) -> list[str]:
    tokens = [token.text for token in nlp(text.lower())]
    return tokens


def query_evaluator(tokens: list[str]) -> None:
    for token in tokens:
        if token in special_tokens:
            pass

