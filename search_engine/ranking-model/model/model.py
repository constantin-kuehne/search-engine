import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_samples, feature_dim = x.size()
        
        x = x.view(
            -1, feature_dim
        )  # Flatten to (batch_size * num_samples, feature_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        x = x.view(batch_size, num_samples)  # Reshape back to (batch_size, num_samples)
        return x
