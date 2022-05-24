import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

UNIT_SIZE = 32
CHART_SIZE = 512

class Window:
    def _on_closing(self):
        self._opened = False

    def __init__(self, w: int, h: int, title: str):
        self._w = w
        self._h = h
        self._root = tk.Tk()
        self._root.geometry(f"{self._w * UNIT_SIZE + CHART_SIZE}x{self._h * UNIT_SIZE}")
        self._canvas = tk.Canvas(self._root, bg = "black", width=self._w*UNIT_SIZE, height=self._h*UNIT_SIZE)
        self._canvas.pack(side=tk.LEFT)
        self._image = np.zeros((self._w, self._h), dtype=np.uint8)
        self._progress = []
        self._progress_dirty = True
        self._opened = True
        self._root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._root.wm_title(title)

        self._figure = plt.Figure(dpi=100)
        self._chart = self._figure.add_subplot(111)
        self._chart_canvas = FigureCanvasTkAgg(self._figure, self._root)
        self._chart_canvas.get_tk_widget().pack(side=tk.RIGHT)


    def set_image(self, image):
        self._image = image

    def set_progress(self, progress: list):
        if self._progress != progress:
            self._progress = progress
            self._progress_dirty = True
    
    def update(self) -> bool:
        self._canvas.delete("all")
        for x in range(self._image.shape[0]):
            for y in range(self._image.shape[1]):
                c = self._image[x, y]
                self._canvas.create_rectangle(
                    x * UNIT_SIZE, 
                    y * UNIT_SIZE, 
                    (x + 1) * UNIT_SIZE,
                    (y + 1) * UNIT_SIZE,
                    fill=f'#{c:02x}{c:02x}{c:02x}',
                )

        if self._progress_dirty:
            self._chart.clear()
            self._chart.plot(self._progress)
            self._chart_canvas.draw()
            self._progress_dirty = False

        self._root.update()
        return self._opened