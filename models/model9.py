from model import Model
import torch
import torch.nn as nn
from torch import optim

# It's no good
class Model9(Model):
    def __init__(self):
        super(Model9, self).__init__()
        self.layers_stack = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(4),
        )

        cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if cuda else "cpu")
        self.to(self._device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        self.criterion = nn.HuberLoss()

    def forward(self, x):
        x = self.layers_stack(x)
        return x
