import random
import cv2
import numpy as np
import colour
import PIL.Image


class ImageGene:
    def __init__(self, image_path, target_image, max_size, mutation_rate) -> None:
        if image_path is not "":
            self.orig_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            self.orig_image = np.zeros((100, 100, 4), dtype=np.uint8)
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
        alpha_channel = image[:, :, 3] / 255
        overlay_colors = image[:, :, :3]
        alpha_mask = alpha_channel[:, :, np.newaxis]

        h, w = image.shape[:2]
        canvas_height, canvas_width = canvas.shape[:2]

        canvas_subsection = canvas[self.y : self.y + h, self.x : self.x + w]

        if self.y + h > canvas_height or self.x + w > canvas_width:
            # Compute safe height and width
            safe_h = min(h, canvas_height - self.y)
            safe_w = min(w, canvas_width - self.x)
            # Extract the subsection of the canvas
            canvas_subsection = canvas[
                self.y : self.y + safe_h, self.x : self.x + safe_w
            ]
            # Adjust the alpha_mask and overlay_colors to match canvas_subsection
            adjusted_alpha_mask = alpha_mask[:safe_h, :safe_w, :]
            adjusted_overlay_colors = overlay_colors[:safe_h, :safe_w, :]

            # Combine the background with the overlay image weighted by alpha
            composite = (
                canvas_subsection * (1 - adjusted_alpha_mask)
                + adjusted_overlay_colors * adjusted_alpha_mask
            )
            # Update the section of the canvas
            canvas[self.y : self.y + safe_h, self.x : self.x + safe_w] = composite
        else:
            composite = canvas_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask
            canvas[self.y : self.y + h, self.x : self.x + w] = composite
        
        return canvas

    def calculate_fitness(self, canvas):
        if canvas is None or self.target_image is None:
            raise ValueError("Canvas or target image is None")
        self.fitness = np.sum(
            colour.difference.delta_e.delta_E_CIE1976(canvas, self.target_image)
        )
        return self.fitness

    def copy(self):
        new_gene = ImageGene("", self.target_image, self.max_size, self.mutation_rate)
        new_gene.orig_image = self.orig_image.copy()
        new_gene.x = self.x
        new_gene.y = self.y
        new_gene.width = self.width
        new_gene.height = self.height
        new_gene.color = self.color.copy()
        new_gene.rotation = self.rotation
        new_gene.opacity = self.opacity
        new_gene.fitness = self.fitness
        return new_gene
