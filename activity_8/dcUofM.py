import numpy as np


def cost(X, y, w):
    """
    Computes cost J = (1 / (2*m)) * sum((h_w(x) - y)^2) 
    """
    m = len(y)
    predictions = hypothesis(X, w)
    cost = (1.0 / (2.0 * m)) * np.sum((predictions - y) ** 2)
    return cost

def hypothesis(X, w):
    """
    Computes h_w(X) = w^T @ X.
    """
    h = np.dot(w.T, X)
    return h

def gradient_descent(X, y, w, alpha, num_iters):
    """
    Iteratively updates weights and biases simultaneously using gradient descent
    """
    m = len(y)
    w = w.astype(float).copy()
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        predictions = hypothesis(X, w)
        errors = predictions - y  # shape: (m,)
        
        # dw_j = (alpha / m) * sum((h_w(x) - y) * x_j)
        dw = (alpha / m) * np.dot(X, errors)
        
        #Simultaneous update of all parameters
        w = w - dw
        
        current_cost = cost(X, y, w)

        # Divergence check: triggers if cost is NaN/Inf or increases above initial cost
        if np.isnan(current_cost) or np.isinf(current_cost) or (i > 0 and current_cost > cost_history[0]):
            print(f"Divergence detected at iteration {i} for alpha={alpha}.")
            cost_history = 'NaN'
            break
        # Track cost at each iteration
        cost_history[i] = current_cost

    return w, cost_history

def llsqr(X, y):
    """
    Analytical Linear Least-Squares
    w = (X @ X^T)^(-1) @ X @ y
    """
    w = np.linalg.inv(np.dot(X, X.T)).dot(X).dot(y)
    return w

def remove_outliers(misfit):
    sigma = np.std(misfit)
    mean_misfit = np.mean(misfit)

    inlier = np.abs(misfit - mean_misfit) <= 2 * sigma
    outlier = ~inlier
    return inlier,outlier


def stdz(x):
    return (x - np.mean(x)) / np.std(x)