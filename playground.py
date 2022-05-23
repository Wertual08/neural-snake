from typing import Tuple
import numpy as np
import random


class Playground:
    def _occupied(self, x: int, y: int) -> bool:
        if hasattr(self, '_snake') and (x, y) in self._snake:
            return True
        if hasattr(self, '_apple') and x == self._apple[0] and y == self._apple[1]:
            return True

        return False 

    def _rand_position(self) -> Tuple[int, int]:
        x = random.randrange(self._w)
        y = random.randrange(self._h)
        while self._occupied(x, y):
            x = random.randrange(self._w)
            y = random.randrange(self._h)

        return x, y

    def _step(self, front: Tuple[int, int]) -> bool:
        if front == self._apple:
            self._apple = self._rand_position()
            self._health = self._w * self._h
            self._visited = set()
        else: 
            self._snake.pop(0)

        if front in self._snake:
            return False
        if front[0] < 0 or front[1] < 0:
            return False
        if front[0] >= self._w or front[1] >= self._h:
            return False

        if front in self._visited:
            self._health -= 1
        self._visited.add(front)

        if self._health <= 0:
            return False

        self._snake.append(front)

        return True

    def __init__(self, w: int, h: int):
        self._w = w
        self._h = h
        self._apple = self._rand_position()
        self._snake = [self._rand_position()]
        self._health = 10
        self._visited = set()

    def move_u(self) -> bool:
        front = self._snake[-1]
        front = (front[0], front[1] + 1)
        return self._step(front)

    def move_l(self) -> bool:
        front = self._snake[-1]
        front = (front[0] - 1, front[1])
        return self._step(front)

    def move_d(self) -> bool:
        front = self._snake[-1]
        front = (front[0], front[1] - 1)
        return self._step(front)

    def move_r(self) -> bool:
        front = self._snake[-1]
        front = (front[0] + 1, front[1])
        return self._step(front)

    def render(self):
        field = np.zeros((self._w, self._h), dtype=np.uint8)
        
        for v in self._visited:
            field[v[0], v[1]] = 20

        field[self._apple[0], self._apple[1]] = 255
        for part in self._snake:
            field[part[0], part[1]] = 50

        if len(self._snake) > 0:
            front = self._snake[-1]
            field[front[0], front[1]] = 50 + self._health * 15

        return field

    def health(self) -> float:
        return self._health / 10

    def score(self) -> int:
        return len(self._snake)
        
        