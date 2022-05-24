from angent import Agent
from playground import Playground
from replay_memory import ReplayMemory
import numpy as np


class Session:
    def __init__(self, w: int, h: int, model, sample: int, target: int, discount: float):
        self._w = w
        self._h = h
        self._playground = Playground(w, h)
        self._memory = ReplayMemory(65536)
        self._agent = Agent(model, self._memory, sample, target, discount)
        self._total_runs = 0
        self._total_steps = 0
        self._total_score = 0
        self._total_reward = 0
        self._max_score = 0

    def update(self, eps: float):
        image = self._playground.render()
        action = self._agent.decide(image)
        
        if np.random.random() < eps:
            action = np.random.randint(0, 4)

        if action == 0:
            alive = self._playground.move_u()
        elif action == 1:
            alive = self._playground.move_l()
        elif action == 2:
            alive = self._playground.move_d()
        else:
            alive = self._playground.move_r()

        if not alive:
            if self._playground.score() > self._max_score:
                self._max_score = self._playground.score()
            self._total_runs += 1
            self._total_steps += self._playground.steps()
            self._total_score += self._playground.score()

            self._playground = Playground(self._w, self._h)
            
        reward = self._playground.health() / 2 + self._playground.bonus() / 2
        self._total_reward += reward

        self._agent.remember(
            image, 
            action, 
            self._playground.render() / 255, 
            reward, 
            1 if alive else 0,
        )

        self._agent.train()

    def finish(self):
        if not self._playground.finished():
            self._total_runs += 1        

    def runs(self) -> int:
        return self._total_runs

    def avg_steps(self) -> float:
        return self._total_steps / self._total_runs

    def avg_score(self) -> float:
        return self._total_score / self._total_runs

    def avg_reward(self) -> float:
        return self._total_reward / self._total_runs

    def reset(self):
        self._playground = Playground(self._w, self._h)
        self._total_runs = 0
        self._total_steps = 0
        self._total_score = 0
        self._total_reward = 0
        self._max_score = 0

    def copy_to(self, target):
        state_dict = self._target_model.state_dict()
        target._model.load_state_dict(state_dict)
        target._target_model.load_state_dict(state_dict)

    def image(self):
        return self._playground.render()

    def save(self, path: str):
        self._agent.save(path)

    def load(self, path: str):
        self._agent.load(path)