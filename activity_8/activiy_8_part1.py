#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:37:32 2026

@author: dcastro
"""

import numpy as np
import matplotlib.pyplot as plt
from dcUofM import hypothesis, cost, gradient_descent, llsqr,remove_outliers


### cofing
alphas = [0.0001,0.001, 0.005, 0.01,0.02]
n_iters = 1500
OUTPUT="FIG"

# ==============================================
# load data 
# ==============================================
data = np.loadtxt('data/ex1data.txt', delimiter=',').T
x_raw = data[0]
y = data[1]
m = len(y)


plt.figure(figsize=(8, 5))
plt.scatter(x_raw, y, color='royalblue', marker='x', label='Training Data')
plt.xlabel('(X)')
plt.ylabel('(y)')
plt.title('Scatter Plot')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig(f"{OUTPUT}/scatterplot.png")
plt.show()

# Add row x_0 with ones 
X = np.vstack((np.ones(m), x_raw))

# ==============================================
# train model 
# ==============================================

#random number crom 0 to 1
np.random.seed(42)
w_init = np.random.rand(X.shape[0])  

cost_zeros = cost(X, y, np.array([0.0, 0.0]))
cost_test = cost(X, y, np.array([-1.0, 2.0]))
print(f"Cost for w=[0, 0]:   {cost_zeros:.4f} (Expected: 32.0727)")
print(f"Cost for w=[-1, 2]:  {cost_test:.4f} (Expected: 54.2400)")



plt.figure(figsize=(8, 5))
final_cost ={}
### Grid Search with fixed alphas
for a in alphas:
    w_cand, cost_hist = gradient_descent(X, y, w_init, a, n_iters)
    final_cost[a] = float(cost_hist[-1])
    plt.plot(range(n_iters), cost_hist, label=f'alpha = {a}')
    
    
best_alpha = min(final_cost, key=final_cost.get)
best_cost = min(final_cost.values())
print (f"Best alpha = {best_alpha} with a cost funct. {best_cost}")

plt.xlabel('Iterations')
plt.ylabel('Cost J(w)')
plt.title('Cost Function for Different Learning Rates (alpha)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{OUTPUT}/cost_func_diff_alphas.png")
plt.show()



w_gd, cost_history = gradient_descent(X, y, w_init, best_alpha, n_iters)

# ==============================================
# prediction and evaluation 
# ==============================================
w_ls = llsqr(X, y)

print(f"\n--- Parameter Comparison alpha={best_alpha} ---")
print(f"Gradient Descent Weights: {w_gd}")
print(f"Least-Squares Weights:    {w_ls}")

# Generate predictions
y_pred_gd = hypothesis(X, w_gd)
y_pred_ls = hypothesis(X, w_ls)

# Plot regression fits
plt.figure(figsize=(8, 5))
plt.scatter(x_raw, y, color='blue', marker='x', label='Observed Data')
plt.plot(x_raw, y_pred_gd, color='red', linewidth=2, label='Gradient Descent Fit')
plt.plot(x_raw, y_pred_ls, color='green', linestyle='--', linewidth=2, label='Least-Squares Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Model Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{OUTPUT}/LRM_comparison.png")
plt.show()

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
plt.savefig(f"{OUTPUT}/misfit.png")
plt.show()


# ==============================================
# Outlier Removal beyond 2*sigma
# ==============================================
# Calculate initial misfit and its standard deviation (sigma)
# least-squares is set as default
misfit = y_pred_ls - y

inlier_mask,outlier_mask=remove_outliers(misfit_ls)
inlier_gd,outlier_gd=remove_outliers(misfit_gd)


X_clean = X[:, inlier_mask]
y_clean = y[inlier_mask]
x_raw_clean = x_raw[inlier_mask]

# Retrain Gradient Descent and Least-Squares on the clean data
w_gd_clean, _ = gradient_descent(X_clean, y_clean, w_init, best_alpha, n_iters)
w_ls_clean = llsqr(X_clean, y_clean)

y_pred_gd_clean = hypothesis(X_clean, w_gd_clean)
y_pred_ls_clean = hypothesis(X_clean, w_ls_clean)

# Compare MSE before and after outlier removal
mse_before = np.mean(misfit ** 2) ##set least-square as default
mse_after_gd = np.mean((y_pred_gd_clean - y_clean) ** 2)
mse_after_ls = np.mean((y_pred_ls_clean - y_clean) ** 2)

print("\n--- Outlier Removal Results ---")
print(f"MSE Before Removal:                 {mse_before:.4f}")
print(f"MSE After Removal (Gradient Desc):  {mse_after_gd:.4f}")
print(f"MSE After Removal (Least-Squares):  {mse_after_ls:.4f}")
print(f"Updated Weights (Least-Squares):    {w_ls_clean}")

# ==============================================
# Plot Comparison Outlier Removal
# ==============================================
plt.figure(figsize=(9, 5))

# Plot kept inliers and removed outliers
plt.scatter(x_raw_clean, y_clean, color='royalblue', marker='x', label='Inliers (Kept)')
plt.scatter(x_raw[outlier_mask], y[outlier_mask], color='red', marker='o', s=70, label='Outliers (> 2σ Removed)')

# Plot regression lines before and after
x_line = np.linspace(np.min(x_raw), np.max(x_raw), 100)
X_line = np.vstack((np.ones(100), x_line))

plt.plot(x_line, hypothesis(X_line, w_ls), color='gray', linestyle='--', linewidth=1.5, label='Original Fit (With Outliers)')
plt.plot(x_line, hypothesis(X_line, w_ls_clean), color='green', linewidth=2, label='Improved Fit (Outliers Removed)')

plt.xlabel('(x)')
plt.ylabel('(y)')
plt.title('Linear Regression (Outlier highlited)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{OUTPUT}/LR_outlier_removal.png")
plt.show()

