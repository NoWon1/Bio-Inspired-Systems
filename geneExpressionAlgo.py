"""
Gene Expression Algorithms (GEA) are inspired by the biological process of gene expression in 
living organisms. This process involves the translation of genetic information encoded in DNA into 
functional proteins. In GEA, solutions to optimization problems are encoded in a manner similar to 
genetic sequences. The algorithm evolves these solutions through selection, crossover, mutation, and 
gene expression to find optimal or near-optimal solutions. GEA is effective for solving complex 
optimization problems in various domains, including engineering, data analysis, and machine 
learning. 
"""

import random 
import math 

POPULATION_SIZE = 50 
GENE_LENGTH = 30 
GENERATIONS = 100 
MUTATION_RATE = 0.05 
CROSSOVER_RATE = 0.7 

TERMINALS = ['x', '1', '2', '3', '4', '5'] 
FUNCTIONS = ['+', '-', '*', '/', 'sin', 'cos'] 

def cost_function(x): 
    return x**2 - 10 * math.sin(2 * x) 

class GeneExpression: 
    def __init__(self): 
        self.gene = self._random_gene() 
        self.cached_fitness = None 

    def _random_gene(self): 
        return [random.choice(TERMINALS + FUNCTIONS) for _ in range(GENE_LENGTH)] 

    def decode_gene(self, x): 
        stack = [] 
        for token in self.gene: 
            if token in TERMINALS: 
                stack.append(float(x) if token == 'x' else float(token)) 
            elif token in FUNCTIONS: 
                if len(stack) >= 1 and token in ['sin', 'cos']: 
                    arg = stack.pop() 
                    stack.append(math.sin(arg) if token == 'sin' else math.cos(arg)) 
                elif len(stack) >= 2: 
                    b, a = stack.pop(), stack.pop() 
                    if token == '+': stack.append(a + b) 
                    elif token == '-': stack.append(a - b) 
                    elif token == '*': stack.append(a * b) 
                    elif token == '/' and b != 0: stack.append(a / b) 
                else: 
                    return float('inf') 
        return stack[0] if len(stack) == 1 else float('inf') 

    def fitness(self, x): 
        if self.cached_fitness is None: 
            try: 
                result = self.decode_gene(x) 
                self.cached_fitness = abs(cost_function(result)) 
            except: 
                self.cached_fitness = float('inf') 
        return self.cached_fitness 

def selection(population, fitnesses): 
    tournament_size = 3 
    candidates = random.sample(list(zip(population, fitnesses)), tournament_size) 
    return min(candidates, key=lambda c: c[1])[0] 

def crossover(parent1, parent2): 
    if random.random() < CROSSOVER_RATE: 
        point = random.randint(1, GENE_LENGTH - 1) 
        child1 = GeneExpression() 
        child2 = GeneExpression() 
        child1.gene = parent1.gene[:point] + parent2.gene[point:] 
        child2.gene = parent2.gene[:point] + parent1.gene[point:] 
        return child1, child2 
    return parent1, parent2 

def mutate(individual): 
    for i in range(GENE_LENGTH): 
        if random.random() < MUTATION_RATE: 
            individual.gene[i] = random.choice(TERMINALS + FUNCTIONS) 

def geneExpression(): 
    population = [GeneExpression() for _ in range(POPULATION_SIZE)] 
    x_value = random.uniform(-10, 10) 

    for generation in range(GENERATIONS): 
        fitnesses = [ind.fitness(x_value) for ind in population] 
        best_idx = fitnesses.index(min(fitnesses)) 
        print(f"Generation {generation}: Best Fitness = {fitnesses[best_idx]:.5f}") 

        new_population = [population[best_idx]] 

        while len(new_population) < POPULATION_SIZE: 
            parent1 = selection(population, fitnesses) 
            parent2 = selection(population, fitnesses) 
            child1, child2 = crossover(parent1, parent2) 
            mutate(child1) 
            mutate(child2) 
            new_population.extend([child1, child2]) 

        population = new_population 

    final_fitnesses = [ind.fitness(x_value) for ind in population] 
    best_idx = final_fitnesses.index(min(final_fitnesses)) 
    print("\nOptimized Solution:") 
    print(f"Best Gene: {population[best_idx].gene}") 
    print(f"Best Fitness: {final_fitnesses[best_idx]:.5f}") 
     
if __name__ == "__main__": 
    geneExpression()
