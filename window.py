import tkinter as tk
import numpy as np

UNIT_SIZE = 32

class Window:
    def _on_closing(self):
        self._opened = False

    def __init__(self, w: int, h: int):
        self._w = w
        self._h = h
        self._root = tk.Tk()
        self._root.geometry(f"{self._w * UNIT_SIZE}x{self._h * UNIT_SIZE}")
        self._canvas = tk.Canvas(self._root, bg = "black", width=self._w*UNIT_SIZE, height=self._h*UNIT_SIZE)
        self._canvas.pack()
        self._image = np.zeros((self._w, self._h), dtype=np.uint8)
        self._opened = True
        self._root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def set_image(self, image):
        self._image = image
    
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
        self._root.update()
        return self._opened