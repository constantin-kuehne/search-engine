import argparse
import array
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class EmbeddingIngestion:
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        truncate_dim: int = 64,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name,
            device="mps" if torch.backends.mps.is_available() else "cpu",
            # model_kwargs={"provider": "CoreMLExecutionProvider"},
        )
        self.model.max_seq_length = 256

        print(f"Loading model {model_name}...")

    def embed_documents_batch(self, documents: list[str]):
        embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def process_corpus(
        self,
        input_tsv: Path,
        output_dir: Path,
        max_docs: int | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = output_dir / "embeddings.npy"
        metadata_path = output_dir / "embedding_metadata"

        print(f"Processing corpus from {input_tsv}")
        print(f"Output directory: {output_dir}")

        print("Counting documents...")

        if max_docs is None:
            with open(output_dir / "document_lengths", "rb") as f:
                file_bytes: int = os.path.getsize(output_dir / "document_lengths")
                document_lengths = array.array("I")
                document_lengths.fromfile(f, file_bytes // 4)
                num_docs = len(document_lengths)
        else:
            num_docs = max_docs

        shape = (num_docs, self.truncate_dim)

        embeddings = np.memmap(
            embeddings_path,
            dtype=np.float32,
            mode="w+",
            shape=shape,
        )

        print(f"Generating embeddings for {num_docs} documents...")
        start_time = time.time()

        batch: list[str] = []
        batch_start_idx = 0
        doc_idx = 0

        with open(input_tsv, "r", encoding="utf-8") as f:
            for line in f:
                if max_docs and doc_idx >= max_docs:
                    break

                row = next(csv.reader([line], delimiter="\t"))
                title = row[2] if len(row) > 2 else ""
                body = row[3] if len(row) > 3 else ""

                batch.append(title + body)
                doc_idx += 1

                if len(batch) >= self.batch_size:
                    batch_embeddings = self.embed_documents_batch(batch)
                    embeddings[batch_start_idx : batch_start_idx + len(batch)] = (
                        batch_embeddings
                    )

                    batch_start_idx += len(batch)
                    batch = []

                    if doc_idx % 10_000 == 0:
                        elapsed = time.time() - start_time
                        docs_per_sec = doc_idx / elapsed
                        print(f"({docs_per_sec:.1f} docs/sec)")

        if batch:
            batch_embeddings = self.embed_documents_batch(batch)
            embeddings[batch_start_idx : batch_start_idx + len(batch)] = (
                batch_embeddings
            )

        embeddings.flush()

        elapsed = time.time() - start_time
        print(f"Finished generating embeddings in {elapsed:.2f}s")
        print(f"Embeddings shape: {embeddings.shape}")

        metadata = {
            "model_name": self.model_name,
            "truncate_dim": self.truncate_dim,
            "num_docs": num_docs,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate document embeddings for semantic search"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to MS MARCO docs TSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=64,
        help="Dimension to truncate embeddings to (Matryoshka)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (optional)",
    )

    args = parser.parse_args()

    ingestion = EmbeddingIngestion(
        model_name=args.model,
        truncate_dim=args.truncate_dim,
        batch_size=args.batch_size,
    )

    ingestion.process_corpus(
        input_tsv=Path(args.input),
        output_dir=Path(args.output),
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()

"""
python -m search_engine.embedding_ingestion \
    --input ./msmarco-docs.tsv \
    --output ./final_embed/ \
    --truncate-dim 384 \
    --batch-size 8 \
    --max-docs 10_000
"""

"""
python -m search_engine.embedding_ingestion \
    --input ./msmarco-docs.tsv \
    --output ./final_embed/ \
    --truncate-dim 384 \
    --batch-size 8 \
    --max-docs 10_000
"""
