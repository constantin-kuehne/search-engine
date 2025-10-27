from search_engine.preprocessing import tokenize_text

POSITIONS = dict[int, list[int]]


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[
            str, tuple[int, POSITIONS]
        ] = {}  # term -> (document frequency, {doc_id: [positions]})

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
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

    def search(self, query: str) -> list[int]:
        tokens = tokenize_text(query)

        doc_list: list[set[int]] = []
        for token in tokens:
            res = self.index.get(token, None)
            print(res)
            if res is not None:
                doc_freq, position_dict = res
                doc_list.append(set(position_dict.keys()))

        if len(doc_list) == 0:
            return []

        if len(doc_list) == 1:
            return list(doc_list[0])

        return list(doc_list[0].intersection(doc_list[1:]))
