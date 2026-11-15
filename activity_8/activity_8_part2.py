#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 22:59:24 2026

@author: dcastro
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part II - Multivariate Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from dcUofM import hypothesis, gradient_descent, llsqr,stdz

### Config
alphas = [0.0001,0.001, 0.005, 0.01,0.02]
#[0.001, 0.005, 0.01, 0.05, 0.1]
n_iters = 1500
OUTPUT="FIG"
# ==============================================
# 1. Load data
# ==============================================
data = np.loadtxt('data/ex1data2.txt', delimiter=',').T
##heather taken from 
#https://github.com/girishkuniyal/Predict-housing-prices-in-Portland.git
x1_raw = data[0]   # Size in square feet 
x2_raw = data[1]   # Number of bedrooms 
y = data[2]        # Price
m = len(y)


x1_std = stdz(x1_raw)
x2_std = stdz(x2_raw)

# Add row x_0 with ones
X = np.vstack((np.ones(m), x1_std, x2_std))

# ==============================================
# Train model & find best alpha
# ==============================================
np.random.seed(42)
w_init = np.random.rand(X.shape[0])

plt.figure(figsize=(8, 5))
final_cost = {}

### Grid Search with fixed alphas
for a in alphas:
    w_cand, cost_hist = gradient_descent(X, y, w_init, a, n_iters)
    final_cost[a] = float(cost_hist[-1])
    plt.plot(range(n_iters), cost_hist, label=f'alpha = {a}')

best_alpha = min(final_cost, key=final_cost.get)
best_cost = min(final_cost.values())
print(f"Best alpha = {best_alpha} with a cost funct. {best_cost:.2f}")

plt.xlabel('Iterations')
plt.ylabel('Cost J(w)')
plt.title('Part II: Cost Function for Different Learning Rates (alpha)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{OUTPUT}/part_2_cost_function.png")
plt.show()

# Train final model over 1,500 iterations
w_gd, cost_history = gradient_descent(X, y, w_init, best_alpha, n_iters)
y_pred_gd = hypothesis(X, w_gd)

# ==============================================
# Evaluate performance on training data
# ==============================================
mse_gd = np.mean((y_pred_gd - y) ** 2)
print(f"\n--- Part II Performance (alpha = {best_alpha}) ---")
print(f"Training MSE (Gradient Descent): {mse_gd:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(range(n_iters), cost_history, color='purple', linewidth=2, label=f'alpha = {best_alpha}')
plt.xlabel('Iterations')
plt.ylabel('Cost J(w)')
plt.title('Part II: Cost Function Over 1,500 Iterations')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{OUTPUT}/part2_cost_function_eval.png")
plt.show()

# Compare weights with Normal Equation (llsqr)
w_ls = llsqr(X, y)
y_pred_ls = hypothesis(X, w_ls)
mse_ls = np.mean((y_pred_ls - y) ** 2)

print(f"\n--- Parameter Comparison alpha={best_alpha} ---")
print(f"Gradient Descent Weights: {w_gd}")
print(f"Normal Equation Weights:  {w_ls}")
print(f"\nMSE (Gradient Descent): {mse_gd:.2f}")
print(f"MSE (Normal Equation):  {mse_ls:.2f}")


# Compute MSE and Misfit (y_hat - y)
mse_gd = np.mean((y_pred_gd - y) ** 2)
mse_ls = np.mean((y_pred_ls - y) ** 2)
print(f"\nMSE (Gradient Descent): {mse_gd:.4f}")
print(f"MSE (Least-Squares):    {mse_ls:.4f}")

misfit_gd = y_pred_gd - y
misfit_ls = y_pred_ls - y

# Plot Misfit Histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].hist(misfit_gd, bins=15, color='salmon', edgecolor='black')
axes[0].set_title('Misfit Histogram: Gradient Descent')
axes[0].set_xlabel('Misfit')
axes[0].set_ylabel('Frequency')

axes[1].hist(misfit_ls, bins=15, color='lightgreen', edgecolor='black')
axes[1].set_title('Misfit Histogram: Least-Squares')
axes[1].set_xlabel('Misfit')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/part_2_misfit.png")
plt.show()
