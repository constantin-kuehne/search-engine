from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import spacy

from search_engine.utils import SearchMode

nlp = spacy.blank("en")


op_precedence = {
    "NOT": 3,
    "-": 3,
    "AND": 2,
    "&": 2,
    "OR": 1,
    "|": 1,
}

op_to_searchmode = {
    "NOT": SearchMode.NOT,
    "-": SearchMode.NOT,
    "AND": SearchMode.AND,
    "&": SearchMode.AND,
    "OR": SearchMode.OR,
    "|": SearchMode.OR,
}


@dataclass
class QueryNode:
    value: str | list[str] | SearchMode
    left: Optional[QueryNode] = None
    right: Optional[QueryNode] = None


def tokenize_text(text: str) -> list[str]:
    tokens = [token.text for token in nlp(text.lower())]
    return tokens


def yard_shunting(tokens: list[str]):
    # algo taken from https://en.wikipedia.org/wiki/Shunting_yard_algorithm
    operator_stack: list[str] = []
    output_queue: list[str | list[str]] = []

    in_phrase: bool = False
    cur_phrase: list[str] = []
    add_and = False

    i = 0

    while i < len(tokens):
        token = tokens[i]

        if in_phrase:
            if token.upper() == '"':
                output_queue.append(cur_phrase)

                in_phrase = False
                cur_phrase = []

            cur_phrase.append(token)
            i += 1
            continue

        if token.upper() == '"':
            in_phrase = True
            i += 1
            continue

        if add_and:
            token = "AND"
            i -= 1
            add_and = False

        if token.upper() in op_precedence.keys():
            while (
                operator_stack
                and (operator_stack[-1] != "(")
                and (
                    (
                        op_precedence[operator_stack[-1].upper()]
                        > op_precedence[token.upper()]
                    )
                    or (
                        (
                            op_precedence[operator_stack[-1].upper()]
                            == op_precedence[token.upper()]
                        )
                        and (token.upper() != "NOT")
                    )
                )
            ):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token.upper())
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output_queue.append(operator_stack.pop())

            if len(operator_stack) == 0:
                raise ValueError("Malformed query. Mismatched parentheses")

            assert operator_stack[-1] == "("
            operator_stack.pop()
        else:
            if len(tokens) - 1 > i and (
                (tokens[i + 1].upper() not in op_precedence.keys())
                and (tokens[i + 1] not in ("(", ")"))
                and (tokens[i + 1] != '"')
            ):
                add_and = True
            output_queue.append(token)
        i += 1

    while operator_stack:
        output_queue.append(operator_stack.pop())

    return output_queue


def build_query_tree(postfix_tokens: list[str | list[str]]):
    stack: list[QueryNode] = []

    for token in postfix_tokens:
        if (type(token) is list) or (token not in op_precedence.keys()):
            stack.append(QueryNode(token))
        else:
            if token == "NOT" or token == "-":
                left = stack.pop()
                right = None
            else:
                right = stack.pop()
                left = stack.pop()

            node = QueryNode(op_to_searchmode[token], left, right)  # type: ignore
            stack.append(node)

    return stack[-1]


if __name__ == "__main__":

    def print_tree(node, indent=0):
        if node:
            print("  " * indent + str(node.value))
            print_tree(node.left, indent + 1)
            print_tree(node.right, indent + 1)

    print_tree(
        build_query_tree(
            yard_shunting(
                ['"', "test", "this", '"', "AND", "(", "test2", "OR", "test3", ")"]
            )
        )
    )
    print("------------")
    print_tree(
        build_query_tree(
            yard_shunting(['"', "test", "this", '"', "AND", "test2", "OR", "test3"])
        )
    )
    print("------------")
    print_tree(
        build_query_tree(yard_shunting(["test", "this", "AND", "test2", "OR", "test3"]))
    )
    print("------------")
    print_tree(
        build_query_tree(
            yard_shunting(["test", "this", "AND", "test2", "OR", "NOT", "test3"])
        )
    )
    print("------------")
    print_tree(build_query_tree(yard_shunting(["test", "and", "test"])))
    print("------------")
    print_tree(build_query_tree(yard_shunting(['"', "test", "and", "test", '"'])))
    print("------------")
    print_tree(
        build_query_tree(
            yard_shunting(tokenize_text('"burj khalifa" AND test OR tower'))
        )
    )
    print("------------")
    print_tree(
        build_query_tree(
            yard_shunting(tokenize_text(')burj test AND test OR tower'))
        )
    )
