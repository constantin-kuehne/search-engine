import torch
import torch.nn.functional as F


def listwise_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    softmax_results = torch.softmax(predictions, dim=1)
    return F.cross_entropy(softmax_results, targets)
