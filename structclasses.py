import math
import random
from typing import List
from dataclasses import dataclass


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

@dataclass
class FitnessData:
    fitness: float
    i: int
    
@dataclass
class MutationsData:
    mutations: List[Mutation]
    indexes: List[int]
    
    def clear(self):
        self.mutations = []
        self.indexes = []
        
    def count(self):
        return len(self.mutations)

@dataclass
class Stats:
    best_fitness: float
    generation: int
    time_for_gen: float

@dataclass
class TriangleCacheData:
    aX: int
    aY: int
    bX: int
    bY: int
    cX: int
    cY: int
    fitness: float
    hash: int

    def data(self):
        return self.fitness

    def equals(self, other):
        tri = other
        return self.aX == tri.aX and self.aY == tri.aY and self.bX == tri.bX and self.bY == tri.bY and self.cX == tri.cX and self.cY == tri.cY

    def hash(self):
        x = self.aX + self.bX + self.cX
        y = self.aY + self.bY + self.cY
        return (97 + x) * 97 + y

    def cached_hash(self):
        return self.hash

    def set_cached_hash(self, hash):
        self.hash = hash