import math
from threading import Thread
from model1 import Model1
from model2 import Model2
from model3 import Model3
from model4 import Model4
from model5 import Model5
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
        iteration += 1
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    session.save(f'dumps/{session.title()}_{timestamp}.torch')
        


sessions = [
    Session(WIDTH, HEIGHT, "model-1", Model1, 64, 256, 0.99, 4096).load('dumps/model-1_20220525091550.torch'),
    Session(WIDTH, HEIGHT, "model-2", Model2, 64, 256, 0.99, 4096).load('dumps/model-2_20220525091553.torch'),
    Session(WIDTH, HEIGHT, "model-3", Model3, 64, 256, 0.99, 4096).load('dumps/model-3_20220525091555.torch'),
    Session(WIDTH, HEIGHT, "model-4", Model4, 64, 256, 0.99, 4096).load('dumps/model-4_20220525091549.torch'),
    Session(WIDTH, HEIGHT, "model-5", Model5, 64, 256, 0.99, 4096).load('dumps/model-5_20220525091551.torch'),
]

threads = [Thread(target=worker, args=(session,)) for session in sessions]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
