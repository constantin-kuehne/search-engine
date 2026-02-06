import torch


def MRR(y_true: torch.Tensor, y_pred: torch.Tensor, num_topk: int = 10) -> torch.Tensor:
    """
    THIS ASSUMES THE FIRST ELEMENT IN y_true IS ALWAYS THE RELEVANT / 1 DOCUMENT AND ALL OTHER ELEMENTS ARE NON-RELEVANT / 0
    y_true: (batch_size, num_samples) tensor of true labels (1 for relevant, 0 for non-relevant)
    y_pred: (batch_size, num_samples) tensor of predicted scores
    """
    positive_scores = y_pred[:, 0:1]
    ranks = (y_pred > positive_scores).sum(dim=1) + 1
    ranks = ranks.float()
    reciprocal_ranks = 1.0 / ranks
    reciprocal_ranks[ranks > num_topk] = 0.0

    return reciprocal_ranks.mean()


def nDCG(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_topk: int = 10
) -> torch.Tensor:
    """
    THIS ASSUMES THE FIRST ELEMENT IN y_true IS ALWAYS THE RELEVANT / 1 DOCUMENT AND ALL OTHER ELEMENTS ARE NON-RELEVANT / 0
    y_true: (batch_size, num_samples) tensor of true labels (1 for relevant, 0 for non-relevant)
    y_pred: (batch_size, num_samples) tensor of predicted scores
    """
    positive_scores = y_pred[:, 0:1]
    ranks = (y_pred > positive_scores).sum(dim=1) + 1
    ranks = ranks.float()
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg[ranks > num_topk] = 0.0
    return dcg.mean()
