import numpy as np
import cv2
from Gene import Gene
from matplotlib import pyplot as plt

# Load target image
target_image_path = "cbpunk.jpg"
target_image = cv2.imread(target_image_path)
org_height, org_width, _ = target_image.shape

MAX_WIDTH = 400
aspect_ratio = org_width / org_height
width = min(org_width, MAX_WIDTH)
weight = int(width / aspect_ratio)

target_image = cv2.resize(target_image, (width, weight))

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)

print("Starting...")

average_color = cv2.mean(target_image)[:3]
canvas = np.full_like(target_image, average_color, dtype=np.uint8)


# Start with an empty list of GDObjects
objects = []
best_fitness = float("inf")  # Start with a very high fitness


def fitness(objects):
    # Create a blank canvas
    canvas = np.full_like(target_image, average_color, dtype=np.uint8)

    # Render each object
    for obj in objects:
        obj.render(canvas)

    difference = cv2.absdiff(target_image, canvas)

    # print(f"Fitness: {np.sum(difference)}")
    # plot the chart of the difference
    # plt.imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
    # plt.show()

    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("Difference", difference)
    # cv2.waitKey(1)

    # Calculate the fitness as the color difference between the target and generated images
    diff = np.sum(difference)
    return diff


# Parameters
generations = 5
mutation_rate = 0.1
REQUIRED_OBJECTS = 5000

objects = [Gene(target_image, mutation_rate) for _ in range(REQUIRED_OBJECTS)]

# fitness(objects)

# exit()

for gen in range(generations):
    """
    1. Create 10 new genes by mutating each of the existing genes
    2. Calculate the fitness of each gene
    3. Select the top 100 genes based on fitness
    4. Repeat
    """

    print(f"Generation {gen}")
    canvas = np.full_like(target_image, average_color, dtype=np.uint8)
    for obj in objects:
        obj.object.render(canvas)

    cv2.imshow("Top Genes", canvas)
    cv2.waitKey(1)

    # 1.
    new_objects = []
    for obj in objects:
        print(obj)
        new_objects.append(obj)
        for _ in range(10):
            new_obj = Gene(target_image, mutation_rate)
            new_obj.object = obj.object.copy()
            new_obj.mutate()
            new_objects.append(new_obj)

    # 2.
    for obj in new_objects:
        obj.fitness = fitness([obj.object])

    # 3.
    new_objects.sort(key=lambda x: x.fitness)
    objects = new_objects[:REQUIRED_OBJECTS]

    # Display the top genes

    # Check if the best gene has improved
    if objects[0].fitness < best_fitness:
        best_fitness = objects[0].fitness
        print(f"Best Fitness: {best_fitness}")


print("Done!")
print("Objects:", objects)


# Display the final result
canvas = np.full_like(target_image, average_color, dtype=np.uint8)
for obj in objects:
    obj.render(canvas)

cv2.imshow("Final Generated Image", canvas)
cv2.imwrite("generated_output.jpg", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
