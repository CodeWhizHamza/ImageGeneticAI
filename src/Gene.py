import random
import cv2
import numpy as np
import os
from typing import Protocol


class Object(Protocol):
    def mutate(self): ...

    def render(self, canvas): ...

    def copy(self): ...

    def crossover(self, other): ...


class Genome:
    def __init__(self, target_image: np.ndarray, mutation_rate: float, max_size: int):
        self.target_image = target_image
        self.mutation_rate = mutation_rate
        self.max_size = max_size
        self.fitness = float("inf")
        self.object = random.choice([Ellipse])(target_image, mutation_rate, max_size)

    def mutate(self):
        self.object.mutate()

    def render(self, canvas):
        self.object.render(canvas)

    def crossover(self, other):
        new_gene = Genome(self.target_image, self.mutation_rate, self.max_size)
        new_ellipse = self.object.crossover(other.object)
        new_gene.object = new_ellipse
        return new_gene

    def __str__(self):
        return f"Gene({self.object})"

    def __refer__(self):
        return self.__str__()


class Ellipse(Object):
    def __init__(self, target_image: np.ndarray, mutation_rate: float, max_size: int):
        self.target_image = target_image
        self.mutation_rate = mutation_rate
        self.max_size = max_size

        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)

        self.size: tuple[int, int] = (
            random.randint(1, self.max_size),
            random.randint(1, self.max_size),
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
                random.randint(1, self.max_size),
                random.randint(1, self.max_size),
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
        new_ellipse = Ellipse(self.target_image, self.mutation_rate, self.max_size)
        new_ellipse.x = int(self.x * 0.5 + other.x * 0.5)
        new_ellipse.y = int(self.y * 0.5 + other.y * 0.5)
        new_ellipse.size = (
            int(self.size[0] * 0.5 + other.size[0] * 0.5),
            int(self.size[1] * 0.5 + other.size[1] * 0.5),
        )
        new_ellipse.color = [
            int(self.color[i] * 0.5 + other.color[i] * 0.5)
            for i in range(len(self.color))
        ]
        new_ellipse.rotation = int(self.rotation * 0.5 + other.rotation * 0.5)
        new_ellipse.opacity = (self.opacity + other.opacity) / 2
        return new_ellipse

    def adapt_mutation_rate(self, fitness_improvement: bool):
        if fitness_improvement:
            self.mutation_rate = max(
                0.01, self.mutation_rate * 0.9
            )  # Reduce rate if improving
        else:
            self.mutation_rate = min(
                0.2, self.mutation_rate * 1.1
            )  # Increase rate if not improving

    def __str__(self):
        return f"Ellipse(x={self.x}, y={self.y}, size={self.size}, color={self.color}, rotation={self.rotation}, opacity={self.opacity})"

    def __refer__(self):
        return self.__str__()


class Rectangle(Object):
    def __init__(self, target_image: np.ndarray, mutation_rate: float, max_size: int):
        self.target_image = target_image
        self.mutation_rate = mutation_rate
        self.max_size = max_size

        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)

        self.size: tuple[int, int] = (
            random.randint(1, self.max_size),
            random.randint(1, self.max_size),
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
                random.randint(1, self.max_size),
                random.randint(1, self.max_size),
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
        cv2.rectangle(
            overlay,
            (self.x, self.y),
            (self.x + self.size[0], self.y + self.size[1]),
            self.color,
            -1,
        )
        cv2.addWeighted(overlay, self.opacity, canvas, 1 - self.opacity, 0, canvas)

    def copy(self):
        new_obj = Rectangle(self.target_image, self.mutation_rate)
        new_obj.x = self.x
        new_obj.y
        new_obj.size = self.size
        new_obj.color = self.color
        new_obj.rotation = self.rotation
        new_obj.opacity = self.opacity
        return new_obj

    def crossover(self, other):
        new_rectangle = Rectangle(self.target_image, self.mutation_rate, self.max_size)
        new_rectangle.x = int(self.x * 0.5 + other.x * 0.5)
        new_rectangle.y = int(self.y * 0.5 + other.y * 0.5)
        new_rectangle.size = (
            int(self.size[0] * 0.5 + other.size[0] * 0.5),
            int(self.size[1] * 0.5 + other.size[1] * 0.5),
        )
        new_rectangle.color = [
            int(self.color[i] * 0.5 + other.color[i] * 0.5)
            for i in range(len(self.color))
        ]
        new_rectangle.rotation = int(self.rotation * 0.5 + other.rotation * 0.5)
        new_rectangle.opacity = (self.opacity + other.opacity) / 2
        return new_rectangle

    def adapt_mutation_rate(self, fitness_improvement: bool):
        if fitness_improvement:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
        else:
            self.mutation_rate = min(0.2, self.mutation_rate * 1.1)


class Triangle(Object):
    def __init__(self, target_image: np.ndarray, mutation_rate: float, max_size: int):
        self.target_image = target_image
        self.mutation_rate = mutation_rate
        self.max_size = max_size

        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)

        self.size: tuple[int, int] = (
            random.randint(1, self.max_size),
            random.randint(1, self.max_size),
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
                random.randint(1, self.max_size),
                random.randint(1, self.max_size),
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
        points = np.array(
            [
                (self.x, self.y),
                (self.x + self.size[0], self.y),
                (self.x + self.size[0] // 2, self.y + self.size[1]),
            ],
            np.int32,
        )
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], self.color)
        cv2.addWeighted(overlay, self.opacity, canvas, 1 - self.opacity, 0, canvas)

    def copy(self):
        new_obj = Triangle(self.target_image, self.mutation_rate)
        new_obj.x = self.x
        new_obj.y = self.y
        new_obj.size = self.size
        new_obj.color = self.color
        new_obj.rotation = self.rotation
        new_obj.opacity = self.opacity
        return new_obj

    def crossover(self, other):
        new_triangle = Triangle(self.target_image, self.mutation_rate, self.max_size)
        new_triangle.x = int(self.x * 0.5 + other.x * 0.5)
        new_triangle.y = int(self.y * 0.5 + other.y * 0.5)
        new_triangle.size = (
            int(self.size[0] * 0.5 + other.size[0] * 0.5),
            int(self.size[1] * 0.5 + other.size[1] * 0.5),
        )
        new_triangle.color = [
            int(self.color[i] * 0.5 + other.color[i] * 0.5)
            for i in range(len(self.color))
        ]
        new_triangle.rotation = int(self.rotation * 0.5 + other.rotation * 0.5)
        new_triangle.opacity = (self.opacity + other.opacity) / 2
        return new_triangle

    def adapt_mutation_rate(self, fitness_improvement: bool):
        if fitness_improvement:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
        else:
            self.mutation_rate = min(0.2, self.mutation_rate * 1.1)


class ImageObject(Object):
    def __init__(
        self, max_size: int, image_path: str, target_image, mutation_rate: float
    ):
        self.image_path = image_path
        # print(image_path)
        self.max_size = max_size
        self.target_image = target_image
        self.mutation_rate = mutation_rate
        self.fitness = float("inf")

        self.image = cv2.imread(image_path)
        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)

        self.size: tuple[int, int] = (
            random.randint(1, self.max_size),
            random.randint(1, self.max_size),
        )

        self.color = target_image[self.y, self.x].tolist()
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.01, 1.0)

    def mutate(self):
        if random.random() < self.mutation_rate:
            self.x = random.randint(0, self.target_image.shape[1] - 1)
            self.y = random.randint(0, self.target_image.shape[0] - 1)

        if random.random() < self.mutation_rate:
            self.size = (
                random.randint(1, self.max_size),
                random.randint(1, self.max_size),
            )

        if random.random() < self.mutation_rate:
            self.color = self.target_image[self.y, self.x].tolist()

        if random.random() < self.mutation_rate:
            self.rotation = random.randint(0, 360)

        if random.random() < self.mutation_rate:
            self.opacity = random.uniform(0.01, 1.0)

    def render(self, canvas):
        # Resize the image to the size of the object
        image = cv2.resize(self.image, self.size)

        # set hue of the image to the color of the object
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image[:, :, 0] = self.color[0]
        image[:, :, 1] = self.color[1]
        image[:, :, 2] = self.color[2]
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # Get the x and y coordinates to place the image
        x1 = self.x - self.size[0] // 2
        x2 = self.x + self.size[0] // 2
        y1 = self.y - self.size[1] // 2
        y2 = self.y + self.size[1] // 2

        # Get the x and y coordinates to place the image
        x1 = max(0, x1)
        x2 = min(canvas.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(canvas.shape[0], y2)

        # Get the x and y coordinates to place the image
        x1_image = 0
        x2_image = x2 - x1
        y1_image = 0
        y2_image = y2 - y1

        x1_image = max(0, x1_image)
        x2_image = min(image.shape[1], x2_image)
        y1_image = max(0, y1_image)
        y2_image = min(image.shape[0], y2_image)

        # Ensure the dimensions are valid
        if x1 < x2 and y1 < y2 and x1_image < x2_image and y1_image < y2_image:
            # Overlay the image on the canvas
            canvas[y1:y2, x1:x2] = cv2.addWeighted(
                image[y1_image:y2_image, x1_image:x2_image],
                self.opacity,
                canvas[y1:y2, x1:x2],
                1 - self.opacity,
                0,
            )

    def copy(self):
        new_obj = ImageObject(
            self.max_size, self.image, self.target_image, self.mutation_rate
        )
        new_obj.x = self.x
        new_obj.y = self.y
        new_obj.size = self.size
        new_obj.color = self.color
        new_obj.rotation = self.rotation
        new_obj.opacity = self.opacity
        return new_obj

    def crossover(self, other):
        new_image_object = ImageObject(
            self.max_size, self.image_path, self.target_image, self.mutation_rate
        )
        new_image_object.x = int(self.x * 0.5 + other.x * 0.5)
        new_image_object.y = int(self.y * 0.5 + other.y * 0.5)
        new_image_object.size = (
            int(self.size[0] * 0.5 + other.size[0] * 0.5),
            int(self.size[1] * 0.5 + other.size[1] * 0.5),
        )
        new_image_object.color = [
            int(self.color[i] * 0.5 + other.color[i] * 0.5)
            for i in range(len(self.color))
        ]
        new_image_object.rotation = int(self.rotation * 0.5 + other.rotation * 0.5)
        new_image_object.opacity = (self.opacity + other.opacity) / 2
        return new_image_object

    def adapt_mutation_rate(self, fitness_improvement: bool):
        if fitness_improvement:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
        else:
            self.mutation_rate = min(0.2, self.mutation_rate * 1.1)

    def __str__(self):
        return f"ImageObject(x={self.x}, y={self.y}, size={self.size}, color={self.color}, rotation={self.rotation}, opacity={self.opacity})"

    def __refer__(self):
        return self.__str__()
