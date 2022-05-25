import math
from threading import Thread
from session import Session
from datetime import datetime
import models


WIDTH = 8
HEIGHT = 8
EPS_START = 0.9
EPS_END = 0.005
EPS_DECAY = 5000


def worker(session: Session):
    session.init_window()

    iteration = 0
    running = True
    while running:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
        running = session.update(eps_threshold)
        iteration += 1
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    session.save(f'dumps/{session.title()}_{timestamp}.torch')
        

sessions = [
    Session(WIDTH, HEIGHT, "model-18", models.Model17, 64, 256, 0.99, 8192),
    Session(WIDTH, HEIGHT, "model-17", models.Model18, 64, 256, 0.99, 8192),
    Session(WIDTH, HEIGHT, "model-19", models.Model19, 64, 256, 0.99, 8192),
    Session(WIDTH, HEIGHT, "model-20", models.Model20, 64, 256, 0.99, 8192),
]

threads = [Thread(target=worker, args=(session,)) for session in sessions]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
