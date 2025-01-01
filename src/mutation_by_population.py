import numpy as np
import cv2
import random

from Gene import Genome

TOTAL_OBJECTS = 5000

# Parameters
population_size = 30
generations = 100
mutation_rate = 0.05

# Load target image
target_image_path = "target_image.jpg"  # Path to your target image
target_image_path = "target_image_eren.jpg"  # Path to your target image
target_image = cv2.imread(target_image_path)
# target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
height, width, _ = target_image.shape

print(f"Image size: {width}x{height}={width*height}")


# Resize target image to a smaller size for faster processing
target_image = cv2.resize(target_image, (width // 4, height // 4))
height, width, _ = target_image.shape

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)


def fitness(objects):
    # Create a blank canvas
    canvas = np.zeros_like(target_image, np.uint8)

    # Render each object
    for obj in objects:
        obj.render(canvas)

    cv2.imshow("Generated Image", canvas)
    cv2.waitKey(1)

    # Calculate fitness as the color difference between target and generated images
    diff = np.sum(np.abs(target_image - canvas))
    return diff


# Initialize population
population = [
    [Genome(target_image, mutation_rate) for _ in range(TOTAL_OBJECTS)]
    for _ in range(population_size)
]  # Each individual has 10 objects

# Evolutionary loop
for gen in range(generations):
    # Evaluate fitness for each individual
    fitness_scores = [fitness(individual) for individual in population]

    # Sort population by fitness (lower is better)
    sorted_population = [
        x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])
    ]

    # Select the top 10% to survive
    survivors = sorted_population[: population_size // 5]

    # Create next generation
    next_gen = []
    for _ in range(population_size):
        # Randomly pick two survivors as parents
        parent1, parent2 = random.sample(survivors, 2)
        child = [random.choice([o1, o2]) for o1, o2 in zip(parent1, parent2)]

        # Mutate child
        for obj in child:
            obj.mutate()
        next_gen.append(child)

    population = next_gen

    # # Display progress
    # if gen % 1 == 0:
    #     print(f"Generation {gen}, Fitness {min(fitness_scores)}")
    # display the generated image
    # best_individual = sorted_population[0]
    # canvas = np.zeros_like(target_image, np.uint8)
    # for obj in best_individual:
    #     obj.render(canvas)

    # cv2.imshow("Generated Image", canvas)
    # # delay for 1ms
    # cv2.waitKey(1)


# Display final result
best_individual = sorted_population[0]
canvas = np.zeros_like(target_image, np.uint8)
for obj in best_individual:
    obj.render(canvas)

cv2.imshow("Generated Image", canvas)
cv2.waitKey(0)

# Save the final result
cv2.imwrite("output.jpg", canvas)

cv2.destroyAllWindows()
