import colour.plotting
import numpy as np
import cv2
import colour
import matplotlib.pyplot as plt
import threading
from helpers import fitness
from ImageGene import ImageGene
import random

random.seed(42)


# Load the target image
target_image = cv2.imread("source_images/cbpunk.jpg")

# Create a canvas to draw the image on
canvas = np.zeros_like(target_image)

# Create an ImageGene object

while True:
    # Render the gene on the canvas
    image_gene = ImageGene(
        "shapes/pokeAPI/underground/star-piece.png", target_image, 100, 0.1
    )
    image_gene.render(canvas)

    # Calculate the fitness of the gene
    fitness_score = fitness(image_gene, canvas, target_image)

    # Display the fitness score
    print(f"Fitness score: {fitness_score}")

    # Display the target image
    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)
