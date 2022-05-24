from pathlib import Path
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        pass

    def feed(self, x, no_grad: bool):
        if no_grad:
            with torch.no_grad():
                return self.forward(x)
        else:
            return self.forward(x)

    def fit(self, result, expected):
        self.optimizer.zero_grad()
        loss = self.criterion(result, expected)
        loss.backward()
        self.optimizer.step()

    def value_to_tensor(self, x):
        return torch.tensor(x, device=self._device)

    def array_to_tensor(self, x):
        return torch.from_numpy(x).float().to(self._device)

    def device(self) -> str:
        return self._device

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()