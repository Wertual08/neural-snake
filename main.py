import math
from session import Session
from window import Window
from datetime import datetime



SESSIONS_COUNT = 1
WIDTH = 8
HEIGHT = 8
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 1000


sessions = [Session(WIDTH, HEIGHT) for _ in range(SESSIONS_COUNT)]
for session in sessions:
    session.load('models/20220524195002.torch')
window = Window(WIDTH, HEIGHT)
iteration = 0

progress = []
while window.update():
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
    for session in sessions:
        session.update(eps_threshold)

    window.set_image(sessions[0].image())

    iteration += 1
    if iteration % 2048 == 0:
        for session in sessions:
            session.finish()

        best = max(sessions, key=lambda x: x.avg_reward())

        progress.append(best.avg_reward())
        print(f"Iteration {iteration}: AVG Steps({best.avg_steps():.3f}), Score({best.avg_score():.3f}) Reward({best.avg_reward():.3f}); eps({eps_threshold})")
        window.set_progress(progress)

        for session in sessions:
            if session != best:
                best.copy_to(session)
            session.reset()



for session in sessions:
    session.finish()
best = max(sessions, key=lambda x: x.avg_reward())
best.save(datetime.now().strftime('models/%Y%m%d%H%M%S.torch'))