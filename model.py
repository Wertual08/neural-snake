from pathlib import Path
import torch
import torch.nn as nn
from torch import optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 1, 2, 1, padding='valid')
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(1, 1, 2, 1, padding='valid')
        self.conv2_act = nn.ReLU()
        self.conv3 = nn.Conv2d(1, 1, 2, 1, padding='valid')
        self.conv3_act = nn.ReLU()
        self.conv4 = nn.Conv2d(1, 1, 2, 1, padding='valid')
        self.conv4_act = nn.ReLU()

        self.layers_stack = nn.Sequential(
            # nn.Conv2d(1, 1, 2, 1, padding='valid'),
            # nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(4),
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
                x1 = self.conv1_act(self.conv1(x))
                x2 = self.conv2_act(self.conv2(x))
                x3 = self.conv3_act(self.conv3(x))
                x4 = self.conv4_act(self.conv4(x))
                x = self.layers_stack(torch.cat([x1, x2, x3, x4], 1))
        else:
            x1 = self.conv1_act(self.conv1(x))
            x2 = self.conv2_act(self.conv2(x))
            x3 = self.conv3_act(self.conv3(x))
            x4 = self.conv4_act(self.conv4(x))
            x = self.layers_stack(torch.cat([x1, x2, x3, x4], 1))
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