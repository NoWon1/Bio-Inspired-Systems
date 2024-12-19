"""
Particle Swarm Optimization (PSO) is inspired by the social behavior of birds flocking or fish 
schooling. PSO is used to find optimal solutions by iteratively improving a candidate solution with 
regard to a given measure of quality. Implement the PSO algorithm using Python to optimize a 
mathematical function. 
"""

import random 

def fitness_function(x): 
    return -x ** 2 

class Particle: 
    def __init__(self, min_x, max_x): 
        self.position = random.uniform(min_x, max_x) 
        self.velocity = random.uniform(-1, 1) 
        self.best_position = self.position 
        self.best_fitness = fitness_function(self.position) 

    def update_velocity(self, global_best_position, inertia, cognitive, social): 
        r1, r2 = random.random(), random.random() 
        cognitive_velocity = cognitive * r1 * (self.best_position - self.position) 
        social_velocity = social * r2 * (global_best_position - self.position) 
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity 

    def update_position(self, min_x, max_x): 
        self.position += self.velocity 
        self.position = max(min(self.position, max_x), min_x) 
        fitness = fitness_function(self.position) 

        if fitness > self.best_fitness: 
            self.best_position = self.position 
            self.best_fitness = fitness 

def particle_swarm_optimization(pop_size, min_x, max_x, inertia, cognitive, social, iterations): 
    swarm = [Particle(min_x, max_x) for _ in range(pop_size)] 
    global_best_position = min(swarm, key=lambda p: p.best_fitness).best_position 
    for iteration in range(iterations): 
        for particle in swarm: 
            particle.update_velocity(global_best_position, inertia, cognitive, social) 
            particle.update_position(min_x, max_x) 
            if fitness_function(particle.position) > fitness_function(global_best_position): 
                global_best_position = particle.position 
        print(f"Iteration {iteration + 1}: Global best = {global_best_position}, Fitness = {fitness_function(global_best_position)}") 
    return global_best_position 

population_size = 30 
min_value = -10 
max_value = 10 
inertia_weight = 0.5 
cognitive_constant = 1.5 
social_constant = 1.5 
num_iterations = 10 

best_solution = particle_swarm_optimization(population_size, min_value, max_value, 
inertia_weight, cognitive_constant, social_constant, num_iterations) 
print(f"Best solution found: {best_solution}, Fitness: {fitness_function(best_solution)}")
