import cv2
import time
import json
import copy
import math
import random
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from algoclasses import *
from structclasses import *
from typing import List, Callable
from dataclasses import dataclass
from scipy.spatial import Delaunay
from concurrent.futures import ThreadPoolExecutor


def trianglesFitness(target: np.ndarray, blockSize: int, n: int):
    h, w, _ = target.shape
    max_diff = 255 * 255 * 3 * w * h

    fitness_functions = []
    for _ in range(n):
        fitness_function = TrianglesImageFitness(
            target_image=target, block_size=blockSize, max_difference=max_diff
        )
        fitness_functions.append(fitness_function)

    return fitness_functions


def generatePoints(n: int) -> NormPointArray:
    points = []
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        points.append(NormPoint(x, y))
    return NormPointArray(points)


def normalize(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    return image / 255.0


with open("./config.json") as f:
    config = json.load(f)


# Generate random points
def pointsFactory() -> NormPointArray:
    return generatePoints(config["points"])


def evaluatorFactory(n: int):
    return ParallelEvaluator(
        trianglesFitness(
            normalized_image.copy(),
            blockSize=5,
            n=n,
        ),
        22,
    )


# Load image
image = cv2.imread(config["image"])
# reduce to half maintaining aspect ratio
image = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)

normalized_image = normalize(image)

mutator = GaussianMethod(0.01, 0.3)
algo = ModifiedGenetic(pointsFactory, 100, 5, evaluatorFactory, mutator)

for _ in range(10):
    algo.step()
    print(algo.get_stats())

    best = algo.get_best()
    points = np.array([[p.x, p.y] for p in best.points], np.float64)
    tri = Delaunay(points)

    for simplex in tri.simplices:
        triangle = points[simplex]
        triangle = (triangle * np.array([image.shape[1], image.shape[0]])).astype(
            np.int32
        )

        A = np.array(triangle[0])
        B = np.array(triangle[1])
        C = np.array(triangle[2])

        # Lengths of the sides
        a = np.linalg.norm(B - C)  # Length of side BC
        b = np.linalg.norm(A - C)  # Length of side AC
        c = np.linalg.norm(A - B)  # Length of side AB

        # Calculate the incenter
        incenter = (a * A + b * B + c * C) / (a + b + c)

        # take color from the original image using the incenter
        incenter_x = min(max(int(incenter[0]), 0), image.shape[1] - 1)
        incenter_y = min(max(int(incenter[1]), 0), image.shape[0] - 1)
        color = (image[incenter_y, incenter_x] / 255).tolist()

        triangle = np.array([triangle], np.int32)
        cv2.fillPoly(
            normalized_image,
            triangle,
            color,
        )

    cv2.imshow("sdfas", normalized_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# cv2.imshow("sdfas", normalized_image)
# cv2.waitKey(0)
