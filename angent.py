from model import Model
from replay_memory import ReplayMemory
import numpy as np
import torch

class Agent:
    def __init__(self, model: Model, target_model: Model, memory: ReplayMemory):
        self._model = model
        self._target_model = target_model
        self._memory = memory
        self._target_counter = 0

        self._target_model.load_state_dict(self._model.state_dict())
        self._target_model.eval()

    def decide(self, states):
        return Model.extract(self._model.forward(self._model.array_to_tensor(states), True))

    def remember(self, states, actions, next_states, rewards):
        for (state, action, next_state, reward) in zip(states, actions, next_states, rewards):
            self._memory.push(
                self._model.array_to_tensor(state), 
                self._model.value_to_tensor(action), 
                self._model.array_to_tensor(next_state), 
                self._model.value_to_tensor(reward),
            )

    def train(self):
        DISCOUNT = 0.99
        SAMPLE_SIZE = 8192

        if len(self._memory) < SAMPLE_SIZE:
            return

        batch = self._memory.sample(SAMPLE_SIZE)
        batch_states = torch.stack(batch.state)
        batch_actions = torch.stack(batch.action)
        batch_next_states = torch.stack(batch.next_state)
        batch_rewards = torch.stack(batch.reward)
        
        decisions = self._model.forward(batch_states, False)
        next_decisions = self._target_model.forward(batch_next_states, True).max(1)[0]

        next_values = next_decisions * DISCOUNT + batch_rewards
        results = decisions.scatter(1, batch_actions.view(-1, 1), next_values.view(-1, 1))

        self._model.fit(decisions, results)

        self._target_counter += 1
        if self._target_counter > 1000:
            self._target_counter = 0
            self._target_model.load_state_dict(self._model.state_dict())

