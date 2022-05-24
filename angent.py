from model import Model
from replay_memory import ReplayMemory
import torch
import numpy as np

class Agent:
    def __init__(self, model: Model, target_model: Model, memory: ReplayMemory):
        self._model = model
        self._target_model = target_model
        self._memory = memory
        self._target_counter = 0

        self._target_model.load_state_dict(self._model.state_dict())
        self._target_model.eval()

    def decide(self, states) -> int:
        return int(self._model.forward(self._model.array_to_tensor(np.array([states])), True).max(1)[1][0])

    def remember(self, state, action, next_state, reward, termination):
        self._memory.push(
            self._model.array_to_tensor(state), 
            self._model.value_to_tensor(action), 
            self._model.array_to_tensor(next_state), 
            self._model.value_to_tensor(reward),
            self._model.value_to_tensor(termination)
        )

    def train(self, sample: int, target: int, discount: float):
        if len(self._memory) < sample:
            return

        batch = self._memory.sample(sample)
        batch_states = torch.stack(batch.state).view(-1, 1, 8, 8)
        batch_actions = torch.stack(batch.action).view(-1, 1)
        batch_next_states = torch.stack(batch.next_state).view(-1, 1, 8, 8)
        batch_rewards = torch.stack(batch.reward)
        batch_terminations = torch.stack(batch.termination)
        
        decisions = self._model.forward(batch_states, False).gather(1, batch_actions)

        target_decisions = self._target_model.forward(batch_next_states, True).max(1)[0]
        next_values = target_decisions * batch_terminations * discount + batch_rewards

        self._model.fit(decisions, next_values.view(-1, 1))

        self._target_counter += 1
        if self._target_counter > target:
            self._target_counter = 0
            self._target_model.load_state_dict(self._model.state_dict())


