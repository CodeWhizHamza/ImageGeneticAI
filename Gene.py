import random
import cv2
import numpy as np
from typing import Protocol


class Object(Protocol):
    max_size = 24

    def mutate(self): ...

    def render(self, canvas): ...

    def copy(self): ...

    def crossover(self, other): ...


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
        # self.color = [random.randint(0, 255) for _ in range(target_image.shape[2])]
        self.color = target_image[self.y, self.x].tolist()
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.01, 1.0)

    def mutate(self):
        if random.random() < self.mutation_rate:
            self.x = random.randint(0, self.target_image.shape[1] - 1)
            self.y = random.randint(0, self.target_image.shape[0] - 1)

        if random.random() < self.mutation_rate:
            self.size = (
                random.randint(1, Ellipse.max_size),
                random.randint(1, Ellipse.max_size),
            )

        if random.random() < self.mutation_rate:
            # self.color = [
            #     random.randint(0, 255) for _ in range(self.target_image.shape[2])
            # ]
            self.color = self.target_image[self.y, self.x].tolist()

        if random.random() < self.mutation_rate:
            self.rotation = random.randint(0, 360)

        if random.random() < self.mutation_rate:
            self.opacity = random.uniform(0.01, 1.0)

    def render(self, canvas):
        overlay = canvas.copy()
        cv2.ellipse(
            overlay,
            (self.x, self.y),
            self.size,
            self.rotation,
            0,
            360,
            self.color,
            -1,
        )
        cv2.addWeighted(overlay, self.opacity, canvas, 1 - self.opacity, 0, canvas)

    def copy(self):
        new_obj = Ellipse(self.target_image, self.mutation_rate)
        new_obj.x = self.x
        new_obj.y = self.y
        new_obj.size = self.size
        new_obj.color = self.color
        new_obj.rotation = self.rotation
        new_obj.opacity = self.opacity
        return new_obj

    def crossover(self, other):
        new_ellipse = Ellipse(self.target_image, self.mutation_rate)
        if random.random() < 0.5:
            new_ellipse.x = self.x
        else:
            new_ellipse.x = other.x

        if random.random() < 0.5:
            new_ellipse.y = self.y
        else:
            new_ellipse.y = other.y

        if random.random() < 0.5:
            new_ellipse.size = self.size
        else:
            new_ellipse.size = other.size

        if random.random() < 0.5:
            new_ellipse.color = self.color
        else:
            new_ellipse.color = other.color

        if random.random() < 0.5:
            new_ellipse.rotation = self.rotation
        else:
            new_ellipse.rotation = other.rotation

        if random.random() < 0.5:
            new_ellipse.opacity = self.opacity
        else:
            new_ellipse.opacity = other.opacity

        return new_ellipse

    def __str__(self):
        return f"Ellipse(x={self.x}, y={self.y}, size={self.size}, color={self.color}, rotation={self.rotation}, opacity={self.opacity})"

    def __refer__(self):
        return self.__str__()


class Gene:
    def __init__(self, target_image: np.ndarray, mutation_rate: float):
        self.target_image = target_image
        self.mutation_rate = mutation_rate
        self.fitness = float("inf")
        self.object = Ellipse(target_image, mutation_rate)

    def mutate(self):
        self.object.mutate()

    def render(self, canvas):
        self.object.render(canvas)

    def crossover(self, other):
        new_gene = Gene(self.target_image, self.mutation_rate)
        new_ellipse = self.object.crossover(other.object)
        new_gene.object = new_ellipse
        return new_gene

    def __str__(self):
        return f"Gene({self.object})"

    def __refer__(self):
        return self.__str__()
