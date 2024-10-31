import random
import cv2
import numpy as np
from typing import Protocol

class Object(Protocol):
    max_size = 50
    def mutate(self):
        pass

    def render(self, canvas):
        pass


class Ellipse(Object):
    def __init__(self, target_image: np.ndarray, mutation_rate: float):
        self.target_image = target_image
        self.mutation_rate = mutation_rate

        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)
        self.size: tuple[int, int] = (random.randint(1, Ellipse.max_size), random.randint(1, Ellipse.max_size))
        self.color = self.target_image[self.y, self.x].tolist()
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.01, 1.0)

    def mutate(self):
        self.x = min(
            max(0, self.x + random.randint(-2, 2)), self.target_image.shape[1] - 1
        )
        self.y = min(
            max(0, self.y + random.randint(-2, 2)), self.target_image.shape[0] - 1
        )
        self.size = tuple(
            min(Ellipse.max_size, max(1, s + random.randint(-5, 5))) for s in self.size
        )
        self.rotation = (self.rotation + random.randint(-360, 360)) % 360
        self.opacity = min(
            1.0, max(0.5, self.opacity + random.uniform(-0.05, 0.05))
        )
        self.color = self.target_image[self.y, self.x].tolist()

    def render(self, canvas):
        color = (
            int(self.color[0] * self.opacity),
            int(self.color[1] * self.opacity),
            int(self.color[2] * self.opacity),
        )

        cv2.ellipse(
            canvas,
            (self.x, self.y),
            self.size,
            self.rotation,
            0,
            360,
            color,
            -1,
        )


class Gene:
    def __init__(self, target_image: np.ndarray, mutation_rate: float):
        self.target_image = target_image
        self.mutation_rate = mutation_rate

        self.object = Ellipse(target_image, mutation_rate)
    
    def mutate(self):
        self.object.mutate()
    
    def render(self, canvas):
        self.object.render(canvas)
    