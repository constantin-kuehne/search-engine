from pathlib import Path

import torch

from .model import RankingModel
from .utils import RankingDataset
from .utils.losses import listwise_loss

file_path = Path(__file__).parent

train_dataset = RankingDataset(
    file_path / "data/train-split.parquet", num_negative_samples=10
)
val_dataset = RankingDataset(
    file_path / "data/val-split.parquet", num_negative_samples=10
)
test_dataset = RankingDataset(
    file_path / "data/test-split.parquet", num_negative_samples=10
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=1
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=1
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=1
)

model = RankingModel(input_dim=10, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10


if __name__ == "__main__":
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            predictions = model(features)
            loss = listwise_loss(predictions, labels)
            print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
