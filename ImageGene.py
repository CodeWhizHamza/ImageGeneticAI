import random
import cv2
import numpy as np
import colour
import PIL.Image


class ImageGene:
    def __init__(self, image_path, target_image, max_size, mutation_rate) -> None:
        self.orig_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.mutation_rate = mutation_rate
        self.max_size = max_size

        self.fitness = float("inf")

        self.target_image = target_image

        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)

        self.width = random.randint(1, target_image.shape[1] - self.x)
        self.height = random.randint(1, target_image.shape[0] - self.y)

        self.color = target_image[self.y, self.x].tolist()
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.01, 1.0)

    def adapt_mutation_rate(self, fitness_improvement: bool):
        if fitness_improvement:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
        else:
            self.mutation_rate = min(0.2, self.mutation_rate * 1.1)

    def mutate(self):
        if random.random() < self.mutation_rate:
            self.x = random.randint(0, self.target_image.shape[1] - 1)
        if random.random() < self.mutation_rate:
            self.y = random.randint(0, self.target_image.shape[0] - 1)
        if random.random() < self.mutation_rate:
            self.width = random.randint(1, self.target_image.shape[1] - self.x)
        if random.random() < self.mutation_rate:
            self.height = random.randint(1, self.target_image.shape[0] - self.y)
        if random.random() < self.mutation_rate:
            self.color = self.target_image[self.y, self.x].tolist()
        if random.random() < self.mutation_rate:
            self.rotation = random.randint(0, 360)
        if random.random() < self.mutation_rate:
            self.opacity = random.uniform(0.01, 1.0)

    def render_gene(self):
        image = self.orig_image.copy()

        # Resize the image to fit within the bounds of the gene
        image = cv2.resize(image, (self.width, self.height))

        # Rotate the image
        image = PIL.Image.fromarray(image)
        image = image.rotate(self.rotation)
        image = np.array(image)

        # Apply the color to the image
        image[:, :, 0] = self.color[0]
        image[:, :, 1] = self.color[1]
        image[:, :, 2] = self.color[2]

        # Apply the opacity to the image
        image[:, :, 3] = (image[:, :, 3] * self.opacity).astype(np.uint8)

        return image

    def render(self, canvas):
        image = self.render_gene()
        # Ensure the dimensions match before rendering to the canvas

        # separate the alpha channel from the color channels
        alpha_channel = image[:, :, 3] / 255
        overlay_colors = image[:, :, :3]

        # To take advantage of the speed of numpy and apply transformations to the entire image with a single operation
        # the arrays need to be the same shape. However, the shapes currently looks like this:
        #    - overlay_colors shape:(width, height, 3)  3 color values for each pixel, (red, green, blue)

        #    - alpha_channel  shape:(width, height, 1)  1 single alpha value for each pixel
        # We will construct an alpha_mask that has the same shape as the overlay_colors by duplicate the alpha channel
        # for each color so there is a 1:1 alpha channel for each color channel
        alpha_mask = alpha_channel[:, :, np.newaxis]

        # The background image is larger than the overlay so we'll take a subsection of the background that matches the
        # dimensions of the overlay.
        # NOTE: For simplicity, the overlay is applied to the top-left corner of the background(0,0). An x and y offset
        # could be used to place the overlay at any position on the background.
        h, w = image.shape[:2]
        canvas_subsection = canvas[self.y : self.y + h, self.x : self.x + w]

        # combine the background with the overlay image weighted by alpha
        composite = canvas_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

        # overwrite the section of the background image that has been updated
        canvas[self.y : self.y + h, self.x : self.x + w] = composite

        return canvas

    def calculate_fitness(self, canvas):
        if canvas is None or self.target_image is None:
            raise ValueError("Canvas or target image is None")
        self.fitness = np.sum(
            colour.difference.delta_e.delta_E_CIE1976(canvas, self.target_image)
        )
        return self.fitness
