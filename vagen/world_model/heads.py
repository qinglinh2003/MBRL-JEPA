import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_mlp: bool = False):
        super().__init__()
        if use_mlp:
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.net = nn.Linear(input_dim, output_dim)

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_tokens)


class RewardHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_tokens).squeeze(-1)
