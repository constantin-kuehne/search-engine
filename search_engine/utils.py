from enum import Enum


class SearchMode(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PHRASE = "PHRASE"
    QUERY_EVALUATOR = "QUERY_EVALUATOR"
