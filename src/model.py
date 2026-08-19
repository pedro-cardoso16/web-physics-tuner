import torch
import torch.nn as nn
from typing import Any


class MLP(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        nn.ParameterDict(
            {
                "k": nn.Parameter(torch.tensor([], dtype=torch.float32)),
            }
        )

        self.layers = nn.Sequential(
            nn.Linear(7, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

if __name__ == "__main__":
    loss = nn.functional.mse_loss
    x = torch.tensor([1],dtype=torch.float32)
    gt = torch.tensor([5],dtype=torch.float32)
    print(loss(x,gt))
    
