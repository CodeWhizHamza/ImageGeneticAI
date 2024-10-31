import numpy as np
import cv2
import random
from Gene import Gene

# Load target image
target_image_path = "target_image_eren.jpg"
target_image = cv2.imread(target_image_path)
height, width, _ = target_image.shape

target_image = cv2.resize(target_image, (width // 4, height // 4))
height, width, _ = target_image.shape

# Parameters
generations = 10000
mutation_rate = 0.1

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)

# Canvas should be initialized with the average color of the target image
average_color = cv2.mean(target_image)[:3]
canvas = np.full_like(target_image, average_color, dtype=np.uint8)


# Start with an empty list of GDObjects
objects = []
best_fitness = float("inf")  # Start with a very high fitness


def fitness(objects):
    # Create a blank canvas
    canvas = np.zeros_like(target_image, np.uint8)

    # Render each object
    for obj in objects:
        obj.render(canvas)

    # Calculate the fitness as the color difference between the target and generated images
    diff = np.sum(np.abs(target_image - canvas))
    return diff


# Evolutionary loop
for gen in range(generations):
    # Add a new object
    new_object = Gene(target_image, mutation_rate)
    objects.append(new_object)

    improved = False  # Track if we find an improvement

    # Try mutations on the new object
    for _ in range(100):  # Limit mutations to avoid infinite loops
        # Mutate the object and calculate fitness
        new_object.mutate()
        new_fitness = fitness(objects)

        # Check if fitness improved
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            improved = True
            break  # Keep this mutation if it improved fitness

    if not improved:
        # If no improvement, remove the latest object
        objects.pop()

    # Display progress
    if gen % 10 == 0:
        print(f"Generation {gen}, Fitness {best_fitness}")
        # Show the current best rendering
        # canvas = np.zeros_like(target_image, np.uint8)
        # Canvas should be initialized with the average color of the target image
        canvas = np.full_like(
            target_image, np.mean(target_image, axis=(0, 1), dtype=np.uint8)
        )

        for obj in objects:
            obj.render(canvas)
        cv2.imshow("Generated Image", canvas)
        cv2.waitKey(1)

# Display the final result
canvas = np.full_like(target_image, average_color, dtype=np.uint8)
for obj in objects:
    obj.render(canvas)

cv2.imshow("Final Generated Image", canvas)
cv2.imwrite("generated_output.jpg", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
