from pathlib import Path

import torch
from tqdm import tqdm

from .model import RankingModel
from .utils.dataset import DevDataset, RankingDataset
from .utils.metrics import MRR, nDCG

file_path = Path(__file__).parent

test_dataset = RankingDataset(
    file_path / "data/test-split.parquet", num_negative_samples=10
)
dev_dataset = DevDataset(file_path / "data/dev.parquet", num_negative_samples=100)

batch_size = 64
num_workers = 4

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

checkpoint_dir = file_path / "checkpoints/1pdz89si"
best_checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
final_checkpoint_path = checkpoint_dir / "last_checkpoint.pth"


if __name__ == "__main__":
    print("==================== BEST CHECKPOINT EVALUATION ====================")
    checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    model = RankingModel(input_dim=10, hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
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

    print(f"Test MRR@10: {avg_test_mrr_10:.4f}")
    print(f"Test nDCG@10: {avg_test_ndcg_10:.4f}")

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

    print(f"Dev MRR@10: {avg_dev_mrr_10:.4f}")
    print(f"Dev nDCG@10: {avg_dev_ndcg_10:.4f}")

    print("==================== LAST CHECKPOINT EVALUATION ====================")
    checkpoint = torch.load(final_checkpoint_path, weights_only=False)
    model = RankingModel(input_dim=10, hidden_dim=checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
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

    print(f"Test MRR@10: {avg_test_mrr_10:.4f}")
    print(f"Test nDCG@10: {avg_test_ndcg_10:.4f}")

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

    print(f"Dev MRR@10: {avg_dev_mrr_10:.4f}")
    print(f"Dev nDCG@10: {avg_dev_ndcg_10:.4f}")
