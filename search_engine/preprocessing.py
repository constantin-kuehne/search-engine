import spacy


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def tokenize_text(text: str) -> list[str]:
    tokens = [token.text for token in nlp(text)]
    return tokens
