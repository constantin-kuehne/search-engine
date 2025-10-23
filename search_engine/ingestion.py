import csv
import nltk


def tokenize_text(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text)
    return tokens


if __name__ == "__main__":
    with open("msmarco-docs.tsv", mode="r") as file:
        reader = csv.DictReader(
            file, delimiter="\t", fieldnames=["docid", "url", "title", "body"]
        )
        for row in reader:
            tokens = tokenize_text(row["body"])
            print(f"Document ID: {row['docid']}, Tokens: {tokens[:10]}...")
