"""
The foraging behavior of ants has inspired the development of optimization algorithms that can solve 
complex problems such as the Traveling Salesman Problem (TSP). Ant Colony Optimization (ACO) 
simulates the way ants find the shortest path between food sources and their nest. Implement the 
ACO algorithm using Python to solve the TSP, where the objective is to find the shortest possible 
route that visits a list of cities and returns to the origin city. 
"""

import numpy as np 
import random 
 
def create_distance_matrix(n_cities): 
    np.random.seed(0) 
    matrix = np.random.randint(1, 100, size=(n_cities, n_cities)) 
    np.fill_diagonal(matrix, 0) 
    return matrix 
 
n_cities = 10 
n_ants = 20 
n_iterations = 50 
alpha = 1 
beta = 2 
evaporation_rate = 0.5 
initial_pheromone = 1 
 
distance_matrix = create_distance_matrix(n_cities) 
pheromone_matrix = np.ones((n_cities, n_cities)) * initial_pheromone 
 
class Ant: 
    def __init__(self, n_cities): 
        self.n_cities = n_cities 
        self.route = [] 
        self.distance_travelled = 0 
 
    def select_next_city(self, current_city, visited): 
        probabilities = [] 
        for city in range(self.n_cities): 
            if city not in visited: 
                pheromone = pheromone_matrix[current_city][city] ** alpha 
                heuristic = (1 / distance_matrix[current_city][city]) ** beta 
                probabilities.append(pheromone * heuristic) 
            else: 
                probabilities.append(0) 
        probabilities = np.array(probabilities) / sum(probabilities) 
        next_city = np.random.choice(range(self.n_cities), p=probabilities) 
        return next_city 
 
    def find_route(self): 
        current_city = random.randint(0, self.n_cities - 1) 
        self.route = [current_city] 
        visited = set(self.route) 
        while len(visited) < self.n_cities: 
            next_city = self.select_next_city(current_city, visited) 
            self.route.append(next_city) 
            self.distance_travelled += distance_matrix[current_city][next_city] 
            visited.add(next_city) 
            current_city = next_city 
        self.distance_travelled += distance_matrix[self.route[-1]][self.route[0]] 
        self.route.append(self.route[0]) 
 
def update_pheromones(ants): 
    global pheromone_matrix 
    pheromone_matrix *= (1 - evaporation_rate) 
    for ant in ants: 
        for i in range(len(ant.route) - 1): 
            city_from = ant.route[i] 
            city_to = ant.route[i + 1] 
            pheromone_matrix[city_from][city_to] += 1.0 / ant.distance_travelled 
            pheromone_matrix[city_to][city_from] += 1.0 / ant.distance_travelled 
 
def ant_colony_optimization(): 
    best_route = None 
    best_distance = float('inf') 
    for iteration in range(n_iterations): 
        ants = [Ant(n_cities) for _ in range(n_ants)] 
        for ant in ants: 
            ant.find_route() 
            if ant.distance_travelled < best_distance: 
                best_distance = ant.distance_travelled 
                best_route = ant.route 
        update_pheromones(ants) 
        print(f"Iteration {iteration + 1}: Best distance = {best_distance}") 
    return best_route, best_distance 
 
best_route, best_distance = ant_colony_optimization() 
print(f"Best route found: {best_route} with distance: {best_distance}")
