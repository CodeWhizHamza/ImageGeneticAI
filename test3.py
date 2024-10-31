import numpy as np
import cv2
import random
import os

# Load target image
target_image_path = "target_image.jpg"
target_image = cv2.imread(target_image_path)
height, width, _ = target_image.shape

# Parameters
generations = 3000  # Large number to evolve over time
mutation_rate = 0.1

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)

# GDObject class to handle predefined shape manipulations
class GDObject:
    shape_folder = "shapes"  # Folder containing shape images

    def __init__(self):
        # Load a random shape from the shapes folder
        self.shape_path = os.path.join(
            GDObject.shape_folder, random.choice(os.listdir(GDObject.shape_folder))
        )

        if not self.shape_path.endswith(".png"):
            return

        self.shape_img = cv2.imread(
            self.shape_path, cv2.IMREAD_UNCHANGED
        )  # Load with alpha channel
        self.size = random.uniform(0.5, 1.0)  # Scale factor for resizing
        self.rotation = random.randint(0, 360)
        self.opacity = random.uniform(0.5, 1.0)

        # Randomize color for each object (R, G, B)
        self.color = [random.randint(0, 255) for _ in range(3)]

        # Position within the target image
        self.x = random.randint(0, target_image.shape[1] - 1)
        self.y = random.randint(0, target_image.shape[0] - 1)

    def apply_color(self, img):
        # Split channels
        b, g, r, a = cv2.split(img)

        # Scale each color channel by the random color values and opacity
        b = (b * self.color[0] / 255).astype(np.uint8)
        g = (g * self.color[1] / 255).astype(np.uint8)
        r = (r * self.color[2] / 255).astype(np.uint8)

        # Merge back with alpha channel and scale alpha by opacity
        colored_img = cv2.merge((b, g, r, a))
        colored_img[..., 3] = (colored_img[..., 3] * self.opacity).astype(np.uint8)
        return colored_img

    def mutate(self):
        # Mutate properties
        if random.random() < mutation_rate:
            self.x = min(max(0, self.x + random.randint(-10, 10)), width - 1)
            self.y = min(max(0, self.y + random.randint(-10, 10)), height - 1)
            self.size = max(0.5, min(1.5, self.size + random.uniform(-0.1, 0.1)))
            self.tint = target_image[self.y, self.x].tolist()
            self.rotation = (self.rotation + random.randint(-10, 10)) % 360
            self.opacity = min(
                1.0, max(0.5, self.opacity + random.uniform(-0.05, 0.05))
            )

    def render(self, canvas):
        # Resize and rotate the shape image
        scaled_shape = cv2.resize(
            self.shape_img,
            (
                int(self.shape_img.shape[1] * self.size),
                int(self.shape_img.shape[0] * self.size),
            ),
        )

        (h, w) = scaled_shape.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
        rotated_shape = cv2.warpAffine(
            scaled_shape, rotation_matrix, (w, h), borderMode=cv2.BORDER_TRANSPARENT
        )

        # Apply color to the shape
        # colored_shape = self.apply_color(rotated_shape)
        colored_shape = rotated_shape

        # Define region of interest on the canvas
        x1, y1 = self.x, self.y
        x2, y2 = x1 + w, y1 + h
        if (
            x2 > canvas.shape[1] or y2 > canvas.shape[0]
        ):  # Ensure the shape fits within canvas bounds
            return

        # Extract ROI from the canvas
        roi = canvas[y1:y2, x1:x2]

        # Blend the colored shape with the canvas based on alpha
        alpha_shape = colored_shape[..., 3] / 255.0  # Normalize alpha channel to 0-1
        for c in range(3):  # Only RGB channels
            roi[..., c] = (
                roi[..., c] * (1 - alpha_shape) + colored_shape[..., c] * alpha_shape
            ).astype(np.uint8)

        # Place blended ROI back into the canvas
        canvas[y1:y2, x1:x2] = roi


def fitness(objects):
    # Create a blank canvas
    canvas = np.zeros_like(target_image, np.uint8)

    # Render each object
    for obj in objects:
        obj.render(canvas)

    # Calculate the fitness as the color difference between the target and generated images
    diff = np.sum(np.abs(target_image - canvas))
    return diff


# Start with an empty list of GDObjects
objects = []
best_fitness = float("inf")  # Start with a very high fitness

# Evolutionary loop
for gen in range(generations):
    # Add a new object
    new_object = GDObject()
    objects.append(new_object)

    improved = False  # Track if we find an improvement

    # Try mutations on the new object
    for _ in range(1):  # Limit mutations to avoid infinite loops
        # Mutate the object and calculate fitness
        new_object.mutate()
        new_fitness = fitness(objects)

        # Check if fitness improved
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            improved = True
            break  # Keep this mutation if it improved fitness

    # for obj in objects:
    #     obj.render(target_image)
    # cv2.imshow("Generated Image", target_image)
    # cv2.waitKey(1)

    if not improved:
        # If no improvement, remove the latest object
        objects.pop()

    # Display progress
    if gen % 10 == 0:
        print(f"Generation {gen}, Fitness {best_fitness}")
        # Show the current best rendering
        canvas = np.zeros_like(target_image, np.uint8)
        for obj in objects:
            obj.render(canvas)
        cv2.imshow("Generated Image", canvas)
        cv2.waitKey(1)

# Display the final result
canvas = np.zeros_like(target_image, np.uint8)
for obj in objects:
    obj.render(canvas)

cv2.imshow("Final Generated Image", canvas)
cv2.imwrite("output.jpg", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
