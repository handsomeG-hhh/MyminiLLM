import torch.nn as nn


class projector(nn.Module):
    def __init__(self, vision_dim=768, hidden_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.projector(x)
