from pathlib import Path
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layers_stack = nn.Sequential(
            # nn.Conv2d(1, 1, 3, 1, padding='valid'),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            # nn.Tanh(),
        )

        cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if cuda else "cpu")
        self.to(self._device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.00005)
        self.criterion = nn.HuberLoss()

    def value_to_tensor(self, x):
        return torch.tensor(x, device=self._device)

    def array_to_tensor(self, x):
        return torch.from_numpy(x).float().to(self._device)

    def forward(self, x, no_grad: bool) -> int:
        if no_grad:
            with torch.no_grad():
                x = self.layers_stack(x)
        else:
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