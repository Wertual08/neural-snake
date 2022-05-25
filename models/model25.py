from model import Model
import torch
import torch.nn as nn
from torch import optim


class Model25(Model):
    def __init__(self):
        super(Model25, self).__init__()
        
        self.conv_pipe = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, (2, 1), 1, padding=(0, 0)),
                nn.ReLU(),
                nn.ZeroPad2d((0, 0, 0, 1)),
            ),
            nn.Sequential(
                nn.Conv2d(1, 1, (1, 2), 1, padding=(0, 0)),
                nn.ReLU(),
                nn.ZeroPad2d((0, 1, 0, 0)),
            ),
            nn.Sequential(
                nn.Conv2d(1, 1, (2, 2), 1, padding=(0, 0)),
                nn.ReLU(),
                nn.ZeroPad2d((0, 1, 0, 1)),
            ),
            nn.Sequential(
                nn.Conv2d(1, 1, (3, 3), 1, padding=(0, 0)),
                nn.ReLU(),
                nn.ZeroPad2d((1, 1, 1, 1)),
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
        x = torch.cat([conv(x) for conv in self.conv_pipe], 1)
        x = self.layers_stack(x)
        return x
