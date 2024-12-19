"""
Cuckoo Search (CS) is a nature-inspired optimization algorithm based on the brood parasitism of 
some cuckoo species. This behavior involves laying eggs in the nests of other birds, leading to the 
optimization of survival strategies. CS uses Lévy flights to generate new solutions, promoting global 
search capabilities and avoiding local minima. The algorithm is widely used for solving continuous 
optimization problems and has applications in various domains, including engineering design, 
machine learning, and data mining. 
"""

import numpy as np 
from scipy.special import gamma 
 
def objective_function(x):  
    return np.sum(x**2) 
 
def levy_flight(alpha=1.5, size=1):  
    sigma_u = np.power((gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / 
                        gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2)), 1 / alpha) 
    u = np.random.normal(0, sigma_u, size) 
    v = np.random.normal(0, 1, size) 
    step = u / np.power(np.abs(v), 1 / alpha) 
    return step 
 
def cuckoo_search(objective_function, n_nests=25, max_iter=1000, pa=0.25):  
    nests = np.random.uniform(low=-5, high=5, size=(n_nests, 2))  
    fitness = np.apply_along_axis(objective_function, 1, nests)  
    best_nest = nests[np.argmin(fitness)] 
    best_fitness = np.min(fitness) 
    for iteration in range(max_iter):  
        for i in range(n_nests):  
            new_nest = nests[i] + levy_flight(size=2)  
            new_fitness = objective_function(new_nest) 
            if new_fitness < fitness[i]:  
                nests[i] = new_nest  
                fitness[i] = new_fitness  
        abandon = np.random.rand(n_nests) < pa 
        nests[abandon] = np.random.uniform(low=-5, high=5, size=(np.sum(abandon), 2))  
        current_best_nest = nests[np.argmin(fitness)]  
        current_best_fitness = np.min(fitness) 
        if current_best_fitness < best_fitness:  
            best_nest = current_best_nest  
            best_fitness = current_best_fitness  
        print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")  
    return best_nest, best_fitness 
 
n_nests = 25  
max_iter = 50  
pa = 0.25  
 
best_solution, best_value = cuckoo_search(objective_function, n_nests, max_iter, pa)  
 
print(f"\nBest solution: {best_solution}")  
print(f"Best fitness value: {best_value}")
