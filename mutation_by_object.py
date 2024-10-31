import colour.plotting
import numpy as np
import cv2
import colour
import matplotlib.pyplot as plt
import threading

from Gene import Gene

# Load target image
target_image_path = "target_image.jpg"
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


def fitness(gene: Gene, canvas: np.ndarray) -> float:
    gene.render(canvas)
    return np.mean(colour.difference.delta_e.delta_E_CIE1976(canvas, target_image))


population_size = 100
parents_count = 100
generations = 1000
mutation_rate = 0.01

print("Creating initial population...")
initial_populations = [
    Gene(target_image, mutation_rate) for _ in range(population_size)
]

average_color = cv2.mean(target_image)[:3]
canvas = np.full_like(target_image, average_color, dtype=np.uint8)

for generation in range(generations):
    print(f"Generation {generation}")

    # Calculate fitness
    threads = []
    for i in range(0, len(initial_populations), 10):
        threads.append(
            threading.Thread(
                target=lambda: [
                    setattr(gene, "fitness", fitness(gene, canvas.copy()))
                    for gene in initial_populations[i : i + 10]
                ]
            )
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    # for i, gene in enumerate(initial_populations):
    #     gene.fitness = fitness(gene, canvas.copy())
    #     print(f"Gene {i}: {gene.fitness}", end="\r")

    # Select parents
    parents = []
    for i in range(parents_count):
        print(f"Selecting parents {i}", end="\r")
        tournament_for_gene_a = np.array(initial_populations)[
            np.random.choice(
                len(initial_populations),
                size=10,
                replace=False,
            )
        ]
        parents.append(min(tournament_for_gene_a, key=lambda x: x.fitness))

    # render parents
    # test_canvas = canvas.copy()
    for parent in parents:
        parent.render(canvas)

    new_population = []

    # TODO: Make crossover such that it improves the fitness
    print("Creating new population")
    for i in range(population_size):
        parent_a = np.random.choice(parents)
        parent_b = np.random.choice(parents)
        child = parent_a.crossover(parent_b)
        child.mutate()
        new_population.append(child)

    cv2.imshow("Canvas", canvas)
    cv2.waitKey(1)
