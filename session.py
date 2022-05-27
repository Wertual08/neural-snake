from angent import Agent
from playground import Playground
from replay_memory import ReplayMemory
import numpy as np

from window import Window


class Session:
    def __init__(self, w: int, h: int, title: str, model, sample: int, target: int, discount: float, memory: int):
        self._w = w
        self._h = h
        self._title = title
        self._playground = Playground(self._w, self._h)
        self._memory = ReplayMemory(memory)
        self._agent = Agent(model, self._memory, sample, target, discount)
        self._total_iterations = 0
        self._total_runs = 0
        self._total_steps = 0
        self._total_score = 0
        self._total_reward = 0
        self._max_score = 0

        self._progress = []

    def init_window(self):
        self._window = Window(self._w, self._h, self._title)

    def update(self, eps: float) -> bool:
        image = self._playground.render()
        
        if np.random.random() < eps:
            action = np.random.randint(0, 4)
        else:
            action = self._agent.decide(image)

        alive = self._playground.move(action)

        target_image = self._playground.render() / 255
            
        reward = self._playground.health() / 2 + self._playground.bonus() / 2
        self._total_reward += reward

        self._agent.remember(
            image, 
            action, 
            target_image, 
            reward, 
            1 if alive else 0,
        )

        self._agent.train()

        if not alive:
            if self._playground.score() > self._max_score:
                self._max_score = self._playground.score()
            self._total_runs += 1
            self._total_steps += self._playground.steps()
            self._total_score += self._playground.score()

            self._playground = Playground(self._w, self._h)

        self._total_iterations += 1
        if self._total_iterations % 2048 == 0:
            self._progress.append([self.avg_steps(), self.avg_score(), self.avg_reward(), eps])
            self._window.set_progress(self._progress)
            self._total_runs = 0
            self._total_score = 0
            self._total_steps = 0
            self._total_reward = 0


        self._window.set_image(self._playground.render())
        return self._window.update()
            

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
        self._agent.copy_to(target._agent)

    def image(self):
        return self._playground.render()

    def save(self, path: str):
        self._agent.save(path)

    def load(self, path: str):
        self._agent.load(path)
        return self

    def title(self) -> str:
        return self._title