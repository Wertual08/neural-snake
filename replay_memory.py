from collections import deque, namedtuple
import random
from typing import List



Transition = namedtuple('Transition', (
    'state',
    'action',
    'next_state',
    'reward',
    'termination',
))

class ReplayMemory:
    def __init__(self, capacity: int):
        self._memory = deque([], capacity)

    def push(self, state, action, next_state, reward, termination):
        self._memory.append(Transition(state, action, next_state, reward, termination))

    def sample(self, size: int) -> Transition:
        return Transition(*zip(*random.sample(self._memory, size)))

    def __len__(self):
        return len(self._memory)