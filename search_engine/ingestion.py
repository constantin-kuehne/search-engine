import csv
import nltk


def tokenize_text(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text)
    return tokens

def process_csv(file_path: str):
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(
            file, delimiter="\t", fieldnames=["docid", "url", "title", "body"]
        )
        for row in reader:
            tokens = tokenize_text(row["body"])
            print(f"Document ID: {row['docid']}, Tokens: {tokens[:10]}...")

if __name__ == "__main__":
    process_csv("../msmarco.csv")
