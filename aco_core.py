import random
import numpy as np
from typing import List
from dataclasses import dataclass
from math import hypot

def euclidean(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

@dataclass
class ACOResult:
    best_route: List[int]
    best_distance: float
    ants: int
    cities: int
    generation_found: int

class ACO:
    def __init__(self, coords, ants=20, alpha=1.0, beta=5.0, rho=0.5, q=100.0, iterations=200, seed=None):
        self.coords = coords
        self.n = len(coords)
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.iterations = iterations
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # distances
        self.dist = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.dist[i,j] = 1e-9
                else:
                    self.dist[i,j] = euclidean(coords[i], coords[j])
        self.pheromone = np.ones((self.n, self.n))
        self.eta = 1.0 / self.dist
        self.best_route = None
        self.best_distance = float('inf')
        self.first_found_generation = None

    def _probabilities(self, current, visited_mask):
        pher = self.pheromone[current] ** self.alpha
        heur = self.eta[current] ** self.beta
        allowed = ~visited_mask
        allowed[current] = False
        numer = pher * heur * allowed
        denom = numer.sum()
        if denom == 0:
            probs = allowed.astype(float)
            probs /= probs.sum()
            return probs
        return numer / denom

    def _build_solution(self):
        tours = []
        lengths = []
        for a in range(self.ants):
            start = random.randrange(self.n)
            tour = [start]
            visited = np.zeros(self.n, dtype=bool)
            visited[start] = True
            current = start
            while len(tour) < self.n:
                probs = self._probabilities(current, visited)
                r = random.random()
                cum = 0.0
                next_city = None
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        next_city = i
                        break
                if next_city is None:
                    choices = [i for i, ok in enumerate(~visited) if ok and i!=current]
                    next_city = random.choice(choices)
                tour.append(next_city)
                visited[next_city] = True
                current = next_city
            length = 0.0
            for i in range(self.n):
                a = tour[i]
                b = tour[(i+1) % self.n]
                length += self.dist[a,b]
            tours.append(tour)
            lengths.append(length)
        return tours, lengths

    def run(self, callback=None):
        for gen in range(1, self.iterations+1):
            tours, lengths = self._build_solution()
            min_idx = int(np.argmin(lengths))
            if lengths[min_idx] < self.best_distance - 1e-12:
                self.best_distance = lengths[min_idx]
                self.best_route = tours[min_idx]
                # Overwrite on every improvement so that, after the loop,
                # first_found_generation holds the generation that first
                # produced the final (ultimate) best route.
                self.first_found_generation = gen
            self.pheromone *= (1.0 - self.rho)
            for tour, L in zip(tours, lengths):
                deposit = self.q / L
                for i in range(self.n):
                    a = tour[i]
                    b = tour[(i+1) % self.n]
                    self.pheromone[a,b] += deposit
                    self.pheromone[b,a] += deposit
            if callback is not None:
                callback(gen, self.best_route, self.best_distance, self.pheromone.copy())
        return ACOResult(best_route=self.best_route or [], best_distance=self.best_distance,
                         ants=self.ants, cities=self.n, generation_found=self.first_found_generation or 0)
