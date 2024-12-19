"""
The Grey Wolf Optimizer (GWO) algorithm is a swarm intelligence algorithm inspired by the social 
hierarchy and hunting behavior of grey wolves. It mimics the leadership structure of alpha, beta, 
delta, and omega wolves and their collaborative hunting strategies. The GWO algorithm uses these 
social hierarchies to model the optimization process, where the alpha wolves guide the search process 
while beta and delta wolves assist in refining the search direction. This algorithm is effective for 
continuous optimization problems and has applications in engineering, data analysis, and machine 
learning.
"""
import numpy as np 
 
def obj_fn(x): 
    return np.sum(x**2) 
 
def gwo(obj_fn, dim, wolves, iters, lb, ub): 
    pos = np.random.uniform(low=lb, high=ub, size=(wolves, dim)) 
    a_pos, b_pos, d_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim) 
    a_score, b_score, d_score = float("inf"), float("inf"), float("inf") 
    
    for t in range(iters): 
        for i in range(wolves): 
            fit = obj_fn(pos[i]) 
            if fit < a_score: 
                d_score, d_pos = b_score, b_pos.copy() 
                b_score, b_pos = a_score, a_pos.copy() 
                a_score, a_pos = fit, pos[i].copy() 
            elif fit < b_score: 
                d_score, d_pos = b_score, b_pos.copy() 
                b_score, b_pos = fit, pos[i].copy() 
            elif fit < d_score: 
                d_score, d_pos = fit, pos[i].copy() 
        
        a = 2 - t * (2 / iters) 
        for i in range(wolves): 
            for j in range(dim): 
                r1, r2 = np.random.rand(), np.random.rand() 
                A1, C1 = 2 * a * r1 - a, 2 * r2 
                D_a = abs(C1 * a_pos[j] - pos[i, j]) 
                X1 = a_pos[j] - A1 * D_a 
 
                r1, r2 = np.random.rand(), np.random.rand() 
                A2, C2 = 2 * a * r1 - a, 2 * r2 
                D_b = abs(C2 * b_pos[j] - pos[i, j]) 
                X2 = b_pos[j] - A2 * D_b 
 
                r1, r2 = np.random.rand(), np.random.rand() 
                A3, C3 = 2 * a * r1 - a, 2 * r2 
                D_d = abs(C3 * d_pos[j] - pos[i, j]) 
                X3 = d_pos[j] - A3 * D_d 
 
                pos[i, j] = (X1 + X2 + X3) / 3 
            
            pos[i] = np.clip(pos[i], lb, ub) 
        
        print(f"Iter {t+1}/{iters}, Best Score: {a_score}, Best Pos: {a_pos}") 
    
    return a_score, a_pos 
 
dim = 5 
wolves = 20 
iters = 20 
lb = -10 
ub = 10 
 
best_score, best_pos = gwo(obj_fn, dim, wolves, iters, lb, ub) 
print("\nFinal Best Score:", best_score) 
print("Final Best Pos:", best_pos)
