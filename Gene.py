import random
import cv2
import numpy as np
from typing import Protocol


class Object(Protocol):
    max_size = 50

    def mutate(self): ...

    def render(self, canvas): ...

    def copy(self): ...


class Ellipse(Object):
    def __init__(self, target_image: np.ndarray, mutation_rate: float):
        self.target_image = target_image
        self.mutation_rate = mutation_rate

        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)
        self.size: tuple[int, int] = (
            random.randint(1, Ellipse.max_size),
            random.randint(1, Ellipse.max_size),
        )
        self.color = [random.randint(0, 255) for _ in range(target_image.shape[2])]
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.01, 1.0)

    def mutate(self):
        self.x = (self.x + random.randint(-10, 10)) % self.target_image.shape[1]
        self.y = (self.y + random.randint(-10, 10)) % self.target_image.shape[0]
        self.size = tuple(
            min(Ellipse.max_size, max(1, s + random.randint(-5, 5))) for s in self.size
        )
        self.rotation = (self.rotation + random.randint(-360, 360)) % 360
        self.opacity = min(1.0, max(0.5, self.opacity + random.uniform(-0.05, 0.05)))
        self.color = [min(255, max(0, c + random.randint(-10, 10))) for c in self.color]

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

    def copy(self):
        new_obj = Ellipse(self.target_image, self.mutation_rate)
        new_obj.x = self.x
        new_obj.y = self.y
        new_obj.size = self.size
        new_obj.color = self.color
        new_obj.rotation = self.rotation
        new_obj.opacity = self.opacity
        return new_obj

    def __str__(self):
        return f"Ellipse(x={self.x}, y={self.y}, size={self.size}, color={self.color}, rotation={self.rotation}, opacity={self.opacity})"

    def __refer__(self):
        return self.__str__()


class Gene:
    def __init__(self, target_image: np.ndarray, mutation_rate: float):
        self.fitness = float("inf")
        self.object = Ellipse(target_image, mutation_rate)

    def mutate(self):
        self.object.mutate()

    def render(self, canvas):
        self.object.render(canvas)

    def __str__(self):
        return f"Gene({self.object})"

    def __refer__(self):
        return self.__str__()
