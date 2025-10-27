import spacy

nlp = spacy.blank("en")


def tokenize_text(text: str) -> list[str]:
    tokens = [token.text for token in nlp(text.lower())]
    return tokens
