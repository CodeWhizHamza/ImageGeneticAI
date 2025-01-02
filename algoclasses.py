import cv2
import time
import copy
import hashlib
import numpy as np

from typing import List
from structclasses import *
from typing import List, Callable
from scipy.spatial import Delaunay
from concurrent.futures import ThreadPoolExecutor

class ModifiedGenetic:
    def __init__(
        self,
        pointsFactory: Callable[[], NormPointArray],
        size: int,
        cutoff: int,
        evaluatorsFactory,
        mutator: GaussianMethod,
    ):
        self.evaluator: ParallelEvaluator = evaluatorsFactory(size)
        self.mutator: GaussianMethod = mutator
        self.population: List[NormPointArray] = [pointsFactory() for _ in range(size)]
        self.new_population = [copy.deepcopy(pg) for pg in self.population]
        self.fitnesses: List[FitnessData] = [
            FitnessData(0, i) for i in range(size)
        ]
        self.mutations: List[List[Mutation]] = [[] for _ in range(size)]
        self.beneficial_mutations: List[MutationsData] = [
            MutationsData(
                mutations=[], indexes=[]
                ) for _ in range(cutoff)
        ]
        self.best: NormPointArray = copy.deepcopy(self.population[0])
        self.cutoff:int = cutoff
        self.stats: Stats = Stats(0, 0, 0)
        self.calculateFitnesses()
        self.updateFitnesses()

    def step(self):
        start_time = time.time()
        self.newGeneration()
        self.calculateFitnesses()
        self.combineMutations()
        self.updateFitnesses()
        self.stats.generation += 1
        self.stats.time_for_gen = time.time() - start_time

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
            bm.mutations.clear()
            bm.indexes.clear()

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
        fit = e.calculate(
            PointsData(points=p, mutations=self.mutations[i])
        )
        self.fitnesses[i].fitness = fit
        self.evaluator.update(i)
        if fit > self.fitnesses[self.get_base(i)].fitness:
            self.set_beneficial(i)

    def set_beneficial(self, index):
        base = self.get_base(index)
        for m in self.mutations[index]:
            found = False
            found_index = -1
            for i, o in enumerate(self.beneficial_mutations[base].mutations): # TODO: Check this !!!!!!!!!!!!!!!!!!!!!!!!!!!!  
                if m.index == o.index:
                    found = True
                    found_index = i
                    break
            if not found:
                self.beneficial_mutations[base].mutations.append(m)
                self.beneficial_mutations[base].indexes.append(index)
            else:
                other = self.beneficial_mutations[base].indexes[found_index]
                if self.fitnesses[index].fitness > self.fitnesses[other].fitness:
                    self.beneficial_mutations[base].mutations[found_index] = m
                    self.beneficial_mutations[base].indexes[found_index] = index

    def combineMutations(self):
        for i in range(len(self.population) - self.cutoff, len(self.population)):
            self.mutations[i] = []
            base = self.get_base(i)
            if len(self.beneficial_mutations[base].mutations) > 0:
                self.population[i] = copy.deepcopy(self.population[base])
                self.evaluator.set_base(i, base)
                for m in self.beneficial_mutations[base].mutations:
                    self.population[i].points[m.index].x = m.new.x
                    self.population[i].points[m.index].y = m.new.y
                    # OLD ^^^^^
                    # self.population[i].points.x = m.new.x
                    # self.population[i].points.y = m.new.y
                    self.mutations[i].append(m)
                e = self.evaluator.get(i)
                fit = e.calculate(
                    PointsData(points=self.population[i], mutations=self.mutations[i])
                )
                self.fitnesses[i].fitness = fit
                self.evaluator.update(i)
            else:
                self.fitnesses[i].fitness = 0

    def updateFitnesses(self):
        self.population = sorted(
            self.population,
            key=lambda x: self.fitnesses[self.population.index(x)].fitness,
        )

        self.best = copy.deepcopy(self.population[0])
        self.stats.best_fitness = self.fitnesses[0].fitness

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
        self.target:np.ndarray = target_image
        self.block_size:int = block_size
        self.max_difference:int = max_difference
        self.triangle_cache: List[TriangleCacheData] = []  # Cache for storing triangle fitness results
        self.next_cache: List[TriangleCacheData] = []  # Cache for storing next generation results
        self.triangulation: CustomDelauany = None

    def calculate(self, pointsData: PointsData) -> float:
        """
        Calculate the fitness of a set of points.
        """
        

        # elif self.base is not None:
        #     # TODO: Check this
        #     self.triangulation = self.base
            
        #     # get the points from the triangulation
        #     previous_points = self.triangulation.simplices
        #     points = np.array([[p.x, p.y] for p in previous_points], np.float64)
        #     # remove only updated points
        #     for mutation in pointsData.mutations:
        #         points[mutation.index] = [mutation.new.x, mutation.new.y]
        #     tri = Delaunay(points).simplices           

        # # Prepare for next generation
        # self.base = None

        # self.next_cache = []

        # difference = 0
        
        # cache_mask = np.zeros((h, w), dtype=np.uint8)
        
        # tri = self.triangle_cache
        
        # area = 0
        # total_area = 0

        # for simplex in self.triangulation.simplices:
        #     pts = self.triangulation.points[simplex]
        #     pts = (pts * np.array([w, h])).astype(np.int32)
        #     triangle_cache = TriangleCacheData(
        #         pts[0][0],
        #         pts[0][1],
        #         pts[1][0],
        #         pts[1][1],
        #         pts[2][0],
        #         pts[2][1],
        #         0,
        #         0,
        #     )
        #     tri_hash = triangle_cache.hash_tri()
        #     triangle_cache.set_cached_hash(tri_hash)

        #     # Check the cache first
        #     for cache in tri:
        #         if cache.cached_hash() == triangle_cache.cached_hash():
        #             difference += cache.data()
        #             break

        #     # If not in cache, calculate
        #     triangle = np.array([pts], np.int32)
        #     mask = np.zeros((h, w), dtype=np.uint8)
        #     cv2.fillPoly(mask, [triangle], 255)

        #     # Calculate triangle area
        #     area = cv2.contourArea(triangle)
        #     total_area += area

        #     # Calculate pixel variance in the triangle
        #     triangle_pixels = self.target[mask == 255]
        #     if len(triangle_pixels) > 0:
        #         mean = np.mean(triangle_pixels, axis=0)
        #         variance = np.mean((triangle_pixels - mean) ** 2)
        #         difference += variance

        #         # Store in cache
        #         triangle_cache.fitness = variance
        #         tri.append(triangle_cache)

        # # Penalty for uncovered pixels
        # blank_area = (h * w) - total_area
        # difference += blank_area * 255

        # return 1 - (difference / self.max_difference)
        
        # not going to use caching for now
        
        h, w, _ = self.target.shape
        points = pointsData.points
        if not isinstance(points, NormPointArray):
            raise TypeError("Expected NormPointArray, got {}".format(type(points)))
        if self.triangulation is None:
            points = np.array([[p.x, p.y] for p in points.points], np.float64)
            simplices = Delaunay(points).simplices
            triangulation = CustomDelauany(points, simplices)
            self.triangulation = triangulation
            
        # TODO PLEASE FIX THIS
        else:
            self.triangulation.points = np.array([[p.x, p.y] for p in points.points], np.float64)
            self.triangulation.simplices = Delaunay(self.triangulation.points).simplices
        
        difference = 0        
        area = 0
        
        for simplex in self.triangulation.simplices:
            pts = self.triangulation.points[simplex]
            pts = (pts * np.array([w, h])).astype(np.int32)
            triangle = np.array([pts], np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [triangle], 255)
            
            # Calculate triangle area
            area = cv2.contourArea(triangle)
            
            # Calculate pixel variance in the triangle
            triangle_pixels = self.target[mask == 255]
            if len(triangle_pixels) > 0:
                mean = np.mean(triangle_pixels, axis=0)
                variance = np.mean((triangle_pixels - mean) ** 2)
                difference += variance
            
            # Penalty for uncovered pixels
            blank_area = (h * w) - area
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
        self.cache: List[TriangleCacheData] = []
        self.next_cache: List[TriangleCacheData] = []

    def get(self, i):
        return self.evaluators[i]

    def prepare(self):
        self.cache, self.next_cache = self.next_cache, self.cache

    def update(self, i: int):
        evaluator = self.evaluators[i]

        # Put triangles calculated from the fitness function into the cache
        # for key, data in evaluator.triangle_cache.copy():
        #     self.cache[key] = data
        for data in evaluator.triangle_cache:
            self.cache.append(data)

        evaluator.set_cache(self.cache)

    def set_base(self, i, base):
        self.evaluators[i].set_base(self.evaluators[base])

    def swap(self, i, j):
        self.evaluators[i], self.evaluators[j] = self.evaluators[j], self.evaluators[i]
