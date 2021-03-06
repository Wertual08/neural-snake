import math
from threading import Thread
from session import Session
from datetime import datetime
import models


WIDTH = 8
HEIGHT = 8
EPS_START = 0 #0.9
EPS_END = 0 #0.005
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
    Session(WIDTH, HEIGHT, "model-25", models.Model25, 64, 512, 0.99, 8192).load('dumps/model-25_20220526125709.torch'),
]

threads = [Thread(target=worker, args=(session,)) for session in sessions]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
