import numpy as np
import math

def computePDF(data_vector, mean_vector, covariance_matrix):
    
    denominator =math.sqrt(((2 * (math.pi)) ** data_vector.size) * np.linalg.det(covariance_matrix))
    difference = data_vector - mean_vector
    covariance_inversed = np.linalg.inv(covariance_matrix)
    mahalanobis_sq = difference.T @ covariance_inversed @ difference
    pwr = - 0.5 * mahalanobis_sq
    numerator = math.exp(pwr)

    return numerator / denominator


def computeLOG(data_vector, mean_vector, covariance_matrix):
    s = data_vector.size
    log_determinant = np.sum(np.log(np.diag(covariance_matrix)))

    difference = data_vector - mean_vector
    
    # Since your covariance matrix is strictly diagonal, 
    # taking the inverse of the diagonal is faster and safer than linalg.inv
    covariance_inversed = np.diag(1.0 / np.diag(covariance_matrix))
    mahalanobis_sq = difference.T @ covariance_inversed @ difference
    
    # Apply the -0.5 multiplier to everything at once
    log_pdf = -0.5 * (s * np.log(2 * np.pi) + log_determinant + mahalanobis_sq)

    return log_pdf
    