import colour.plotting
import numpy as np
import cv2
import colour
import matplotlib.pyplot as plt
import threading
from helpers import fitness

from Gene import Genome

import os

if not os.path.exists("records"):
    os.makedirs("records")
if not os.path.exists("plots"):
    os.makedirs("plots")

# Load target image
target_image_path = "source_images/crumble.jpg"
target_image = cv2.imread(target_image_path)
org_height, org_width, _ = target_image.shape

MAX_WIDTH = 400
aspect_ratio = org_width / org_height
width = min(org_width, MAX_WIDTH)
weight = int(width / aspect_ratio)

print(f"Resizing image to {width}x{weight}")

target_image = cv2.resize(target_image, (width, weight))

cv2.imshow("Target Image", target_image)
cv2.waitKey(0)

print("Starting...")

population_size = 1000
parents_count = 100
generations = 40000
mutation_rate = 0.2

print("Creating initial population...")
initial_populations = [
    Genome(target_image, mutation_rate, 30) for _ in range(population_size)
]

canvas = np.full_like(target_image, (126, 126, 126), dtype=np.uint8)

last_fitness = float("-inf")

fitness_vs_generation = []

for generation in range(generations):
    print(f"Generation {generation}")

    # Calculate fitness
    threads = []
    items_per_thread = 80
    for i in range(0, len(initial_populations), items_per_thread):
        threads.append(
            threading.Thread(
                target=lambda: [
                    setattr(
                        gene,
                        "fitness",
                        fitness(gene, canvas.copy(), target_image.copy()),
                    )
                    for gene in initial_populations[i : i + items_per_thread]
                ]
            )
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    # for i, gene in enumerate(initial_populations):
    #     gene.fitness = fitness(gene, canvas.copy(), target_image.copy()))
    #     print(f"Gene {i}: {gene.fitness}", end="\r")

    # Select parents
    parents = []
    for i in range(parents_count):
        print(f"Selecting parents {i}", end="\r")
        tournament_for_gene_a = np.array(initial_populations)[
            np.random.choice(
                len(initial_populations),
                size=parents_count,
                replace=False,
            )
        ]
        parents.append(min(tournament_for_gene_a, key=lambda x: x.fitness))

    # render parents
    # test_canvas = canvas.copy()
    for parent in parents:
        parent.render(canvas)

    current_fitness = np.sum(colour.difference.delta_E_CIE1976(canvas, target_image))

    new_population = []

    # TODO: Make crossover such that it improves the fitness
    print("Creating new population")
    for i in range(population_size):
        parent_a = np.random.choice(parents)
        parent_b = np.random.choice(parents)
        child = parent_a.crossover(parent_b)
        child.object.adapt_mutation_rate(current_fitness > last_fitness)
        child.mutate()
        new_population.append(child)

    # cv2.imshow("Canvas", canvas)
    # cv2.waitKey(1)

    if generation % 50 == 0:
        fitness_vs_generation.append((generation, current_fitness))

    if generation % 100 == 0:
        print(f"Generation {generation} Fitness: {current_fitness}")
        cv2.imwrite(f"records/generated_{generation}.png", canvas)

        plt.plot(*zip(*fitness_vs_generation))
        plt.savefig(f"plots/fitness_generation_{generation}.png")

    last_fitness = current_fitness
