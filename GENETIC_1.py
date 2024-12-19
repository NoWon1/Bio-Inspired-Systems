"""
Genetic Algorithms (GA) are inspired by the process of natural selection and genetics, where the 
fittest individuals are selected for reproduction to produce the next generation. GAs are widely used 
for solving optimization and search problems. Implement a Genetic Algorithm using Python to solve 
a basic optimization problem, such as finding the maximum value of a mathematical function.
"""

import random

def fitness_function(x):
    return x ** 2

def generate_individual(min_x, max_x):
    return random.uniform(min_x, max_x)

def generate_population(pop_size, min_x, max_x):
    return [generate_individual(min_x, max_x) for _ in range(pop_size)]

def selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    return random.choices(population, weights=selection_probs, k=2)

def crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

def mutation(individual, mutation_rate, min_x, max_x):
    if random.random() < mutation_rate:
        return generate_individual(min_x, max_x)
    return individual

def genetic_algorithm(pop_size, min_x, max_x, generations, mutation_rate):
    population = generate_population(pop_size, min_x, max_x)
    for generation in range(generations):
        fitness_scores = [fitness_function(ind) for ind in population]
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate, min_x, max_x)
            child2 = mutation(child2, mutation_rate, min_x, max_x)
            new_population.extend([child1, child2])
        population = new_population
        best_individual = max(population, key=fitness_function)
        print(f"Generation {generation + 1}: Best solution = {best_individual}, Fitness = {fitness_function(best_individual)}")
    return max(population, key=fitness_function)

population_size = 20
min_value = -10
max_value = 10
num_generations = 10
mutation_probability = 0.1

best_solution = genetic_algorithm(population_size, min_value, max_value, num_generations, mutation_probability)
print(f"Best solution found: {best_solution}, Fitness: {fitness_function(best_solution)}")
