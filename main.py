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
print(model)
agent = Agent(
    model,
    Model(),
    ReplayMemory(65536 * 2),
)


PLAYGROUNDS_COUNT = 1
WIDTH = 8
HEIGHT = 8

max_score = 0
playgrounds = [Playground(WIDTH, HEIGHT) for _ in range(PLAYGROUNDS_COUNT)]


window = Window(WIDTH, HEIGHT)

iteration = 0
while window.update():
    images = [playground.render() / 255 for playground in playgrounds]
    decisions = agent.decide(np.stack(images))

    actions = []
    next_images = []
    rewards = []
    
    for i, decision in enumerate(decisions):
        playground = playgrounds[i]
        action = np.argmax(decision)

        if np.random.random() < 0.05:
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
            reward = 0
            if playground.score() > max_score:
                max_score = playground.score()
                print(f"Max score: {max_score}")
            playgrounds[i] = Playground(WIDTH, HEIGHT)
        else:
            reward = playground.health()

        actions.append(action)
        next_images.append(playground.render() / 255)
        rewards.append(reward)

    agent.remember(images, actions, next_images, rewards)

    agent.train()

    window.set_image(playgrounds[0].render())

model.save(datetime.now().strftime('models/%Y%m%d%H%M%S.torch'))