import numpy as np
import cv2
import random
import colour
import os
import shutil

from Gene import Gene

# Load target image
image_name = "me_and_ahmed"
target_image_path = f"{image_name}.jpg"

os.makedirs(image_name, exist_ok=True)


target_image = cv2.imread(target_image_path)
height, width, _ = target_image.shape

TARGET_WIDTH = 400

if width > TARGET_WIDTH:
    target_image = cv2.resize(
        target_image, (TARGET_WIDTH, int(height * TARGET_WIDTH / width))
    )

width, height, _ = target_image.shape

# Parameters
generations = 50000  # Large number to evolve over time
mutation_rate = 0.1

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)


def fitness(canvas, object):
    # Render the object on the canvas
    object.render(canvas)

    return np.sum(colour.difference.delta_E_CIE1976(canvas, target_image))


# Start with an empty list of GDObjects
canvas = np.zeros_like(target_image, np.uint8)
best_fitness = float("inf")  # Start with a very high fitness

max_size = 100

# Evolutionary loop
for gen in range(generations):
    # Add a new object
    obj1 = Gene(target_image, mutation_rate, max_size)
    # objects.append(new_object)

    improved = False  # Track if we find an improvement

    # Try mutations on the new object
    for _ in range(20):  # Limit mutations to avoid infinite loops
        # Mutate the object and calculate fitness
        obj1.mutate()
        new_fitness = fitness(canvas.copy(), obj1)

        # Check if fitness improved
        if new_fitness < best_fitness:
            best_fitness = new_fitness
            improved = True
            break  # Keep this mutation if it improved fitness

    if improved:
        obj1.render(canvas)

    # I want to change the maximum size of the object over time
    if gen % 100 == 0:
        max_size = max(10, max_size - random.randint(-1, 2))

    # Display progress
    if gen % 10 == 0:
        print(f"Generation {gen}, Fitness {best_fitness}")
        # Show the current best rendering
        # canvas = np.zeros_like(target_image, np.uint8)
        # for obj in objects:
        #     obj.render(canvas)
        cv2.imshow("Generated Image", canvas)
        cv2.waitKey(1)

    if gen % 250 == 0:
        # Save the current best rendering
        cv2.imwrite(f"{image_name}/output_{gen}.jpg", canvas)


# Display the final result
# canvas = np.zeros_like(target_image, np.uint8)
# for obj in objects:
#     obj.render(canvas)

cv2.imshow("Final Generated Image", canvas)
cv2.imwrite("output.jpg", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
