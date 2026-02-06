import polars as pl
import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [
    "bm25_score",
    "bm25_score_body",
    "bm25_score_title",
    "body_first_occurrence_mean",
    "title_first_occurrence_mean",
    "body_first_occurrence_min",
    "title_first_occurrence_min",
    "body_length_norm",
    "title_length_norm",
    "in_title",
]

MAX_NEGATIVE_SAMPLES = 10


class RankingDataset(Dataset):
    def __init__(self, parquet_path, num_negative_samples: int = 4):
        self.lf = pl.read_parquet(parquet_path)
        if num_negative_samples > MAX_NEGATIVE_SAMPLES:
            raise ValueError("num_negative_samples should not exceed 10")

        self.num_negative_samples = (
            num_negative_samples + 2
        )  # +2 to account for matches in negative samples with the postivive sample (one for each hard and easy)

    def _create_neg_example_names(self, num_negative_samples, match: int | None = None):
        half = (
            num_negative_samples // 2
            if num_negative_samples % 2 == 0
            else (num_negative_samples // 2) + 1
        )
        if match is None:
            increasing = list(range(1, half))
            decreasing = list(range(100, 100 - (num_negative_samples - half - 1), -1))
            return increasing + decreasing

        if match <= half:
            increasing = list(range(1, half + 1))
            increasing.remove(match)
            decreasing = list(range(100, 100 - (num_negative_samples - half - 1), -1))
            return increasing + decreasing

        increasing = list(range(1, half))
        decreasing = list(range(100, 100 - (num_negative_samples - half), -1))
        decreasing.remove(match)
        return increasing + decreasing

    def __len__(self):
        return len(self.lf)

    def _extract_data(self, row: pl.DataFrame) -> torch.Tensor:
        neg_example_names = self._create_neg_example_names(
            self.num_negative_samples, row["match"].to_list()[0]
        )
        pos_values = [row[col].to_list()[0] for col in FEATURE_COLUMNS]
        neg_values = [
            [row[f"{col}_{neg}"].to_list()[0] for col in FEATURE_COLUMNS]
            for neg in neg_example_names
        ]

        sample = torch.tensor([pos_values] + neg_values)
        return sample

    def __getitem__(self, idx):
        row = self.lf[idx]
        sample = self._extract_data(row).squeeze(-1)
        target = torch.tensor([1.0] + [0.0] * (self.num_negative_samples - 2), dtype=torch.float)
        return sample, target
