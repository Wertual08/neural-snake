from pathlib import Path
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv_pipe = []
        for _ in range(4):
            self.conv_pipe.append(nn.Sequential(
                nn.Conv2d(1, 1, 3, 1, padding='valid'),
                nn.ReLU(),
            ))
        for _ in range(5):
            self.conv_pipe.append(nn.Sequential(
                nn.Conv2d(1, 1, 2, 1, padding='valid'),
                nn.ReLU(),
            ))

        self.layers_stack = nn.Sequential(
            # nn.Conv2d(1, 1, 2, 1, padding='valid'),
            # nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(4),
            # nn.Tanh(),
        )

        cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if cuda else "cpu")
        self.to(self._device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.000025)
        self.criterion = nn.HuberLoss()

    def value_to_tensor(self, x):
        return torch.tensor(x, device=self._device)

    def array_to_tensor(self, x):
        return torch.from_numpy(x).float().to(self._device)

    def forward(self, x, no_grad: bool) -> int:
        if no_grad:
            with torch.no_grad():
                x = torch.cat([conv(x) for conv in self.conv_pipe], 1)
                x = self.layers_stack(x)
        else:
            x = torch.cat([conv(x) for conv in self.conv_pipe], 1)
            x = self.layers_stack(x)
        return x

    @staticmethod
    def extract(x):
        return x.cpu().detach().numpy()

    def fit(self, result, expected):
        self.optimizer.zero_grad()
        loss = self.criterion(result, expected)
        loss.backward()
        self.optimizer.step()

    def device(self) -> str:
        return self._device

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()