import cv2
import numpy as np
import json
from typing import List, Callable
import time
import copy
from dataclasses import dataclass
from typing import List
import random
from scipy.spatial import Delaunay
from concurrent.futures import ThreadPoolExecutor
import hashlib
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt


@dataclass
class NormPoint:
    x: float
    y: float

    def constrain(self):
        self.x = max(0, min(self.x, 1))
        self.y = max(0, min(self.y, 1))

    def distance_to(self, other: "NormPoint") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def copy(self):
        return NormPoint(self.x, self.y)


@dataclass
class Mutation:
    old: NormPoint
    new: NormPoint
    index: int


@dataclass
class NormPointArray:
    points: List[NormPoint]

    def set(self, points: List[NormPoint]):
        if len(points) == len(self.points):
            self.points = points
        else:
            raise ValueError("[PointGroup] Length of points must be the same")

    def constrain(self):
        for p in self.points:
            p.constrain()

    def copy(self):
        return NormPointArray([p.copy() for p in self.points])


@dataclass
class PointsData:
    points: NormPointArray
    mutations: List[Mutation]


@dataclass
class GaussianMethod:
    rate: float
    amount: float

    def mutate(self, points: NormPointArray, callback):
        for i, point in enumerate(points.points):
            new_x = point.x + random.gauss(0, self.amount)
            new_y = point.y + random.gauss(0, self.amount)
            new_point = NormPoint(new_x, new_y)
            new_point.constrain()
            callback(Mutation(point, new_point, i))


class ModifiedGenetic:
    def __init__(
        self,
        pointsFactory: Callable[[], NormPointArray],
        size: int,
        cutoff: int,
        evaluatorsFactory,
        mutator: GaussianMethod,
    ):
        self.population = [pointsFactory() for _ in range(size)]
        self.new_population = [copy.deepcopy(pg) for pg in self.population]
        self.best: NormPointArray = copy.deepcopy(self.population[0])
        self.evaluator = evaluatorsFactory(size)
        self.fitnesses = [{"index": i, "fitness": 0} for i in range(size)]
        self.mutations: List[List[Mutation]] = [[] for _ in range(size)]
        self.beneficial_mutations: List[dict] = [
            {"mutations": [], "indexes": []} for _ in range(cutoff)
        ]
        self.mutator = mutator
        self.cutoff = cutoff
        self.stats: dict = {"generation": 0, "time_for_gen": 0, "best_fitness": 0}
        self.calculateFitnesses()
        self.updateFitnesses()

    def step(self):
        start_time = time.time()
        self.newGeneration()
        self.calculateFitnesses()
        self.combineMutations()
        self.updateFitnesses()
        self.stats["generation"] += 1
        self.stats["time_for_gen"] = time.time() - start_time

    def newGeneration(self):
        i = 0
        for i in range(self.cutoff):
            self.new_population[i] = copy.deepcopy(self.population[i])
            self.mutations[i] = []

        while i < len(self.population) - self.cutoff:
            for j in range(self.cutoff):
                if i >= len(self.population) - self.cutoff:
                    break
                self.mutations[i] = []
                self.new_population[i] = copy.deepcopy(self.population[j])
                self.evaluator.set_base(i, j)
                self.mutator.mutate(
                    self.new_population[i], lambda mut: self.mutations[i].append(mut)
                )
                i += 1

        for bm in self.beneficial_mutations:
            bm["mutations"].clear()
            bm["indexes"].clear()

        self.population, self.new_population = self.new_population, self.population

    def calculateFitnesses(self):
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(self.population) - self.cutoff):
                p = self.population[i]
                e = self.evaluator.get(i)
                futures.append(executor.submit(self._calculate_fitness, i, p, e))
            for future in futures:
                future.result()
        self.evaluator.prepare()

    def _calculate_fitness(self, i, p, e):
        fit = e.calculate({"points": p, "mutations": self.mutations[i]})
        self.fitnesses[i]["fitness"] = fit
        self.evaluator.update(i)
        if fit > self.fitnesses[self.get_base(i)]["fitness"]:
            self.set_beneficial(i)

    def set_beneficial(self, index):
        base = self.get_base(index)
        for m in self.mutations[index]:
            found = False
            found_index = -1
            for i, o in enumerate(self.beneficial_mutations[base]["mutations"]):
                if m["index"] == o["index"]:
                    found = True
                    found_index = i
                    break
            if not found:
                self.beneficial_mutations[base]["mutations"].append(m)
                self.beneficial_mutations[base]["indexes"].append(index)
            else:
                other = self.beneficial_mutations[base]["indexes"][found_index]
                if self.fitnesses[index]["fitness"] > self.fitnesses[other]["fitness"]:
                    self.beneficial_mutations[base]["mutations"][found_index] = m
                    self.beneficial_mutations[base]["indexes"][found_index] = index

    def combineMutations(self):
        for i in range(len(self.population) - self.cutoff, len(self.population)):
            self.mutations[i] = []
            base = self.get_base(i)
            if len(self.beneficial_mutations[base]["mutations"]) > 0:
                self.population[i] = copy.deepcopy(self.population[base])
                self.evaluator.set_base(i, base)
                for m in self.beneficial_mutations[base]["mutations"]:
                    self.population[i][m["index"]]["x"] = m["new"]["x"]
                    self.population[i][m["index"]]["y"] = m["new"]["y"]
                    self.mutations[i].append(m)
                e = self.evaluator.get(i)
                fit = e.calculate(
                    {"points": self.population[i], "mutations": self.mutations[i]}
                )
                self.fitnesses[i]["fitness"] = fit
                self.evaluator.update(i)
            else:
                self.fitnesses[i]["fitness"] = 0

    def updateFitnesses(self):
        self.population = sorted(
            self.population,
            key=lambda x: self.fitnesses[self.population.index(x)]["fitness"],
        )

        self.best = copy.deepcopy(self.population[0])
        self.stats["best_fitness"] = self.fitnesses[0]["fitness"]

    def get_base(self, index):
        return index % self.cutoff

    def get_best(self):
        return self.best

    def get_stats(self):
        return self.stats


class TrianglesImageFitness:
    def __init__(
        self, target_image: np.ndarray, block_size: int, max_difference: float
    ):
        self.target = target_image
        self.block_size = block_size
        self.max_difference = max_difference
        self.triangle_cache: dict = {}  # Cache for storing triangle fitness results

    def _hash_triangle(self, points):
        """
        Create a hash for a triangle based on its points.
        """
        return hashlib.md5(points.tobytes()).hexdigest()

    def calculate(self, pointsData: dict) -> float:
        """
        Calculate the fitness of a set of points.
        """
        h, w, _ = self.target.shape
        points = pointsData.get("points")
        if not isinstance(points, NormPointArray):
            raise TypeError("Expected NormPointArray, got {}".format(type(points)))

        points = np.array([[p.x, p.y] for p in points.points], np.float64)

        # Triangulate the points
        tri = Delaunay(points)
        difference = 0
        total_area = 0

        for simplex in tri.simplices:
            pts = points[simplex]
            triangle_hash = self._hash_triangle(pts)

            # Check the cache first
            if triangle_hash in self.triangle_cache:
                difference += self.triangle_cache[triangle_hash]
                continue  # Skip recalculating

            # If not in cache, calculate
            triangle = np.array([pts], np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [triangle], 255)

            # Calculate triangle area
            area = cv2.contourArea(triangle)
            total_area += area

            # Calculate pixel variance in the triangle
            triangle_pixels = self.target[mask == 255]
            if len(triangle_pixels) > 0:
                mean = np.mean(triangle_pixels, axis=0)
                variance = np.mean((triangle_pixels - mean) ** 2)
                difference += variance

                # Store in cache
                self.triangle_cache[triangle_hash] = variance

        # Penalty for uncovered pixels
        blank_area = (h * w) - total_area
        difference += blank_area * 255

        return 1 - (difference / self.max_difference)

    def set_cache(self, cache):
        self.triangle_cache = cache

    def set_base(self, base):
        self.triangle_cache = base.triangle_cache


class ParallelEvaluator:
    def __init__(
        self, fitness_funcs: List[TrianglesImageFitness], cache_power_of_2: int
    ):
        self.evaluators = fitness_funcs
        self.cache: dict = {}
        self.next_cache: dict = {}

    def get(self, i):
        return self.evaluators[i]

    def prepare(self):
        self.cache, self.next_cache = self.next_cache, self.cache

    def update(self, i: int):
        evaluator = self.evaluators[i]

        # Put triangles calculated from the fitness function into the cache
        for key, data in evaluator.triangle_cache.items():
            self.cache[key] = data

        evaluator.set_cache(self.cache)

    def set_base(self, i, base):
        self.evaluators[i].set_base(self.evaluators[base])

    def swap(self, i, j):
        self.evaluators[i], self.evaluators[j] = self.evaluators[j], self.evaluators[i]


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


# Load image
image = cv2.imread(config["image"])
normalized_image = normalize(image)


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


mutator = GaussianMethod(0.01, 0.3)
algo = ModifiedGenetic(pointsFactory, 400, 5, evaluatorFactory, mutator)


algo.step()

best = algo.get_best()
points = np.array([[p.x, p.y] for p in best.points], np.float64)
tri = Delaunay(points)

# plot the best image
for simplex in tri.simplices:
    triangle = points[simplex] * 255
    triangle = np.array([triangle], np.int32)
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.fillPoly(normalized_image, triangle, color)


cv2.imshow("sdfas", normalized_image)
cv2.waitKey(0)
