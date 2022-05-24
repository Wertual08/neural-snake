import math
from threading import Thread
from model1 import Model1
from session import Session
from window import Window
from datetime import datetime



WIDTH = 8
HEIGHT = 8
EPS_START = 0.9
EPS_END = 0.001
EPS_DECAY = 10000


def worker(session: Session):
    session.init_window()

    iteration = 0
    running = True
    while running:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
        running = session.update(eps_threshold)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    session.save(f'models/{session.title()}_{timestamp}.torch')
        


sessions = [
    Session(WIDTH, HEIGHT, "model-1", Model1, 64, 256, 0.99),#.load('models/20220524224029.torch'),
]

threads = [Thread(target=worker, args=(session,)) for session in sessions]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
