from model import Model
import torch
import torch.nn as nn
from torch import optim


class Model2(Model):
    def __init__(self):
        super(Model2, self).__init__()
        
        self.conv_pipe = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(1, 1, 2, 1, padding='valid'),
                nn.ReLU(),
            ) for _ in range(8)]
        )

        self.layers_stack = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(392),
            nn.ReLU(),
            nn.LazyLinear(392),
            nn.ReLU(),
            nn.LazyLinear(4),
        )

        cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if cuda else "cpu")
        self.to(self._device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        self.criterion = nn.HuberLoss()

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.conv_pipe], 1)
        x = self.layers_stack(x)
        return x
