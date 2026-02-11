from pathlib import Path

import torch
from tqdm import tqdm

import wandb

from .model import RankingModel
from .utils.dataset import DevDataset, RankingDataset
from .utils.losses import listwise_loss
from .utils.metrics import MRR, nDCG

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

dev_dataset = DevDataset(file_path / "data/dev.parquet", num_negative_samples=100)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

lr = 5e-4
config = {
    "input_dim": 10,
    "hidden_dim": 256,
}

model = RankingModel(**config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 25


if __name__ == "__main__":
    run = wandb.init(
        project="search-engine-ranking",
        config={
            **config,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        },
    )
    best_checkpoint_path = file_path / f"checkpoints/{run.id}/best_checkpoint.pth"
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = file_path / f"checkpoints/{run.id}/last_checkpoint.pth"
    last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_metric = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch in pbar:
            features, labels = batch
            predictions = model(features)
            loss = listwise_loss(predictions, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            wandb.log(
                {"train/loss_step": loss.item(), "train/global_step": global_step}
            )
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_loader)

        wandb.log({"train/loss_epoch": avg_epoch_loss, "epoch": epoch})

        model.eval()
        val_mrr = 0.0
        val_ndcg = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                features, labels = batch
                predictions = model(features)
                val_loss = listwise_loss(predictions, labels)
                val_loss += val_loss.item()

                val_mrr += MRR(labels, predictions).item()
                val_ndcg += nDCG(labels, predictions).item()

        avg_val_mrr = val_mrr / len(val_loader)
        avg_val_ndcg = val_ndcg / len(val_loader)
        avg_val_loss = val_loss / len(val_loader)
        mean_val_metric = (avg_val_mrr + avg_val_ndcg) / 2

        wandb.log(
            {
                "val/MRR@10": avg_val_mrr,
                "val/nDCG@10": avg_val_ndcg,
                "val/mean_metric": mean_val_metric,
                "val/loss": avg_val_loss,
                "epoch": epoch,
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}, Val MRR: {avg_val_mrr:.4f}, Val nDCG: {avg_val_ndcg:.4f}"
        )

        if mean_val_metric > best_metric:
            best_metric = mean_val_metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    **config,
                },
                best_checkpoint_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            **config,
        },
        last_checkpoint_path,
    )

    print("\n" + "=" * 50)
    print("Running final evaluation on test dataset...")
    print("=" * 50)

    model.eval()
    test_mrr_10 = 0.0
    test_ndcg_10 = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            features, labels = batch
            predictions = model(features)

            test_mrr_10 += MRR(labels, predictions, num_topk=10).item()
            test_ndcg_10 += nDCG(labels, predictions, num_topk=10).item()

    avg_test_mrr_10 = test_mrr_10 / len(test_loader)
    avg_test_ndcg_10 = test_ndcg_10 / len(test_loader)

    print("\n" + "=" * 50)
    print("FINAL TEST SET RESULTS")
    print("=" * 50)
    print(f"MRR@10:  {avg_test_mrr_10:.4f}")
    print(f"nDCG@10: {avg_test_ndcg_10:.4f}")
    print("=" * 50)

    wandb.log({"test/MRR@10": avg_test_mrr_10, "test/nDCG@10": avg_test_ndcg_10})

    print("\n" + "=" * 50)
    print("Running final evaluation on dev dataset...")
    print("=" * 50)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    model.eval()
    dev_mrr_10 = 0.0
    dev_ndcg_10 = 0.0

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating on dev set"):
            features, labels = batch
            predictions = model(features)

            dev_mrr_10 += MRR(labels, predictions, num_topk=10).item()
            dev_ndcg_10 += nDCG(labels, predictions, num_topk=10).item()

    avg_dev_mrr_10 = dev_mrr_10 / len(dev_loader)
    avg_dev_ndcg_10 = dev_ndcg_10 / len(dev_loader)

    print("\n" + "=" * 50)
    print("FINAL DEV SET RESULTS")
    print("=" * 50)
    print(f"MRR@10:  {avg_dev_mrr_10:.4f}")
    print(f"nDCG@10: {avg_dev_ndcg_10:.4f}")
    print("=" * 50)

    wandb.log({"dev/MRR@10": avg_dev_mrr_10, "dev/nDCG@10": avg_dev_ndcg_10})

    wandb.finish()
