import math
import time
from angent import Agent
from playground import Playground
from model import Model
from PIL import Image
import numpy as np
from replay_memory import ReplayMemory
from window import Window
from datetime import datetime


model = Model()
model.load('models/20220523231554.torch')
print(model)
agent = Agent(
    model,
    Model(),
    ReplayMemory(16384),
)


PLAYGROUNDS_COUNT = 1
WIDTH = 8
HEIGHT = 8

max_score = 0
playgrounds = [Playground(WIDTH, HEIGHT) for _ in range(PLAYGROUNDS_COUNT)]


window = Window(WIDTH, HEIGHT)

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

iteration = 0
total_sessions = 0
total_steps = 0
total_score = 0
total_reward = 0
while window.update():
    images = [playground.render() / 255 for playground in playgrounds]
    decisions = agent.decide(np.stack(images))
    
    actions = []
    next_images = []
    rewards = []
    terminations = []
    
    for i, action in enumerate(decisions):
        playground = playgrounds[i]

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
        if np.random.random() < eps_threshold:
            action = np.random.randint(0, 4)

        if action == 0:
            alive = playground.move_u()
        elif action == 1:
            alive = playground.move_l()
        elif action == 2:
            alive = playground.move_d()
        else:
            alive = playground.move_r()

        if not alive:
            if playground.score() > max_score:
                max_score = playground.score()
            total_sessions += 1
            total_steps += playground.steps()
            total_score += playground.score()

            playgrounds[i] = Playground(WIDTH, HEIGHT)
            
        reward = playground.health()
        total_reward += reward

        actions.append(action)
        next_images.append(playground.render() / 255)
        rewards.append(reward)
        terminations.append(1 if alive else 0)

    agent.remember(images, actions, next_images, rewards, terminations)

    agent.train()

    window.set_image(playgrounds[0].render())

    iteration += 1
    if iteration % 1024 == 0:
        total_steps /= total_sessions
        total_score /= total_sessions
        total_reward /= total_sessions 
        print(f"Iteration {iteration}: AVG Steps({total_steps:.3f}), Score({total_score:.3f}), Reward({total_reward:.3f}); MAX Score({max_score}); eps({eps_threshold})")
        max_score = 0
        total_sessions = 0
        total_steps = 0
        total_score = 0


model.save(datetime.now().strftime('models/%Y%m%d%H%M%S.torch'))