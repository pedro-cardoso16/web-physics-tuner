import torch
import torch.nn as nn
import json

from typing import Any

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """Train Dataset

    Used to extract train data from sythetic data with the known hyperparameters.

    """

    def __init__(self, file: str) -> None:
        super().__init__()

        with open(file) as f:
            data = json.load(f)

            n_nodes: int = len(data["frames"][0]["nodes"])
            n_frames: int = len(data["frames"])  # total number of frames
            n_data = (n_nodes * n_frames) - 1
            frames = data["frames"]

            self.data = torch.empty((n_data, 7), dtype=torch.float32)
            self.label = torch.tensor([n_data, 2], dtype=torch.float32)
            self.hyperparams = data["hyperparams"]

            for index in range(n_data):
                frame_index = index // n_nodes
                frame_node_index = index % n_nodes
                frame = frames[frame_index]
                next_frame = frames[frame_index + 1]

                node_position = frame["nodes"][frame_node_index]
                node_velocity = frame["velocity"][frame_node_index]

                self.data[index, 0] = node_position[0]  # x position
                self.data[index, 1] = node_position[1]  # y position
                self.data[index, 2] = node_velocity[0]  # x velocity
                self.data[index, 3] = node_velocity[1]  # y velocity
                self.data[index, 4] = (
                    frame_node_index + 1
                ) / n_nodes  # node proportion [0,1]
                self.data[index, 5] = (n_nodes - 1) / 100  # number of nodes [0,1]
                self.data[index, 6] = next_frame["dt"]  # dt into future frame [0,0.05]

                # label data
                next_node_position = next_frame["nodes"][frame_node_index]
                self.label[index, 0] = next_node_position[0]
                self.label[index, 1] = next_node_position[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        return self.data[index], self.label[index]


class MLP(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        nn.ParameterDict(
            {
                "k": nn.Parameter(torch.tensor(0.0, dtype=torch.float32)),
                "dr": nn.Parameter(torch.tensor(0.0, dtype=torch.float32)),
                "g": nn.Parameter(torch.tensor(0.0, dtype=torch.float32)),
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
    x = torch.tensor([1], dtype=torch.float32)
    gt = torch.tensor([5], dtype=torch.float32)

    model = MLP()
    # Training process
    for epoch in range(10):
        pass

        output = model()

    print(loss(x, gt))
