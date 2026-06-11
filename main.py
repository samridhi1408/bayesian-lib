import numpy as np
from foundation.multiGaussian import computePDF
import pandas as pd
from EM import expectMaximize

def generate_data(n_points=300):
    """Creates three random clusters to test the algorithm."""
    # Cluster 1: Mean [1, 1], Cluster 2: Mean [4, 4], Cluster 3: Mean [9, 9]
    c1 = np.random.randn(n_points // 3, 2) + np.array([1, 1])
    c2 = np.random.randn(n_points // 3, 2) + np.array([4, 4])
    c3 = np.random.randn(n_points // 3, 2) + np.array([9, 9])
    return np.vstack([c1, c2, c3]).astype(np.float64)

data_vector = np.array([170.0, 70.0])

mean_vector = np.array([165, 65])

covariance_matrix = np.eye(2)

result = computePDF(data_vector, mean_vector, covariance_matrix)
print(result)

data = generate_data()
final_means, final_covs, final_priors = expectMaximize(data, k=3, iterations=100)

print("\nFinal Means found by EM:")
print(final_means)

