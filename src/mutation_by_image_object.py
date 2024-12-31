import numpy as np
import cv2
import colour
import matplotlib.pyplot as plt
import threading
from helpers import fitness
import random
import time

from ImageObject import ImageObject

import os

if not os.path.exists("records"):
    os.makedirs("records")
if not os.path.exists("plots"):
    os.makedirs("plots")

# Load target image
target_image_path = "source_images/cbpunk.jpg"
target_image = cv2.imread(target_image_path)
org_height, org_width, _ = target_image.shape

MAX_WIDTH = 400
aspect_ratio = org_width / org_height
width = min(org_width, MAX_WIDTH)
weight = int(width / aspect_ratio)

print(f"Resizing image to {width}x{weight}")

target_image = cv2.resize(target_image, (width, weight))

cv2.imshow("Target Image", target_image)
cv2.waitKey(3)

print("Starting...")

# load shapes
shapes = os.listdir("shapes/pokeAPI/underground")
shapes = [f"shapes/pokeAPI/underground/{shape}" for shape in shapes]

populations = 1000  # number of populations to go through evolutions using generations
generations = 10 # number of generations to go through evolutions using parents

population_size = 10  # number of genes in a population
parents_count = 100  # number of parents to select from a population
child_mutation_rate = 0.8  # mutation rate for a child gene
mutation_rate = 0.3  # mutation rate for a gene

initial_population = []

best_genes = []

canvas = np.full_like(target_image, (126, 126, 126), dtype=np.uint8)

last_fitness = float("inf")

fitness_vs_generation = []
final_fitness_vs_generation = []

for population in range(populations):

    print(f"Population {population}")
    initial_population = []
    for member in range(population_size):
        # random shape from the shapes
        random.seed(time.time())
        shape = random.choice(shapes)
        initial_population.append(ImageObject(shape, target_image, MAX_WIDTH, mutation_rate))

    for generation in range(generations):
        print(f"Generation {generation}")

        # Calculate fitness
        def calculate_fitness_thread(gene, canvas_copy):
            gene.calculate_fitness(canvas_copy)

        threads = []
        for gene in initial_population:
            canvas_copy = canvas.copy()
            thread = threading.Thread(
                target=calculate_fitness_thread, args=(gene, canvas_copy)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # sort the population by fitness
        initial_population.sort(key=lambda x: x.fitness)

        # Display the best gene
        best_gene = initial_population[0]

        print(f"Best gene fitness: {best_gene.fitness}")

        # Select top parents
        parents = initial_population[:parents_count]

        # Create new population
        new_population = []
        for parent in parents:
            new_population.append(parent)
            parent.adapt_mutation_rate(parent.fitness < last_fitness)
            for _ in range(9):
                child = parent.copy()
                child.mutate(child_mutation_rate)
                new_population.append(child)

        initial_population = []
        initial_population = new_population
        print(f"New Population size: {len(initial_population)}")

    # Display the best gene
    best_gene = initial_population[0]

    print(f"Best gene fitness: {best_gene.fitness}")
    fitness_vs_generation.append(best_gene.fitness)

    last_fitness = best_gene.fitness
    final_fitness_vs_generation.append(last_fitness)

    # Render the best gene on the canvas
    best_gene.render(canvas)

    best_genes.append(best_gene)

    # Display the canvas

    # display the best gene
    display_canvas = np.full_like(target_image, (126, 126, 126), dtype=np.uint8)
    best_gene.render(display_canvas)
    cv2.imwrite(f"records/best_gene{population}.png", display_canvas)

    cv2.imshow("canvas", canvas)
    cv2.imwrite(f"records/{population}.png", canvas)
    # plot both the fitnesses
    fig, ax = plt.subplots(2)
    ax[0].plot(fitness_vs_generation)
    ax[0].set_title("Fitness vs Generation")
    ax[1].plot(final_fitness_vs_generation)
    ax[1].set_title("Final Fitness vs Generation")
    plt.close(fig)
    plt.savefig(f"plots/plot.png")
    cv2.waitKey(1)

cv2.imshow("canvas", canvas)
plt.plot(final_fitness_vs_generation)
plt.savefig("plots/best_gene.png")
cv2.waitKey(0)

new_canvas = np.full_like(target_image, (126, 126, 126), dtype=np.uint8)

# sort the best genes by fitness
best_genes.sort(key=lambda x: x.fitness)

for gene in range(300):
    best_genes[gene].render(new_canvas)

cv2.imshow("canvas", new_canvas)
cv2.waitKey(0)

cv2.imwrite("records/best_gene.jpg", canvas)

print("Done")
