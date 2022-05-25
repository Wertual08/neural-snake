from model import Model
import torch
import torch.nn as nn
from torch import optim


class Model23(Model):
    def __init__(self):
        super(Model23, self).__init__()
        
        self.conv_pipe = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, (3, 1), 1, padding=(0, 0)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1, 1, (3, 1), 1, padding=(0, 0)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1, 1, (1, 3), 1, padding=(0, 0)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1, 1, (1, 3), 1, padding=(0, 0)),
                nn.ReLU(),
            ),
        ])

        self.layers_stack = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(4),
        )

        cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if cuda else "cpu")
        self.to(self._device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        self.criterion = nn.HuberLoss()

    def forward(self, x):
        x = torch.cat([conv(x) if i < len(self.conv_pipe) // 2 else conv(x).transpose(-2, -1) for i, conv in enumerate(self.conv_pipe)], 1)
        x = self.layers_stack(x)
        return x
