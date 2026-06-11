import numpy as np
from foundation.multiGaussian import computePDF

def expectMaximize(data, k = 3, iterations = 500):
    points, features = data.shape

    means = data[np.random.choice(points, k, replace = False)]
    covariances = [np.eye(features) for _ in range (k)]

    # considering circly is equally likely at the start
    priors = np.ones(k) / k

    weights = np.zeros((points, k))

    for i in range(iterations):
        # E
        for n in range(points):
            for j in range(k):
                weights[n,j] = priors[j] * computePDF(data[n], means[j], covariances[j])

            total_weight = np.sum(weights[n, :])
            if total_weight > 0:
                weights[n, :] /= total_weight

        # M
        sum_weights = np.sum(weights, axis=0)

        for j in range(k):
            priors[j] = sum_weights[j] / points

            means[j] = np.sum(weights[:, [j]] * data, axis=0) / sum_weights[j]

            diff = data - means[j]

            covariances[j] = (weights[:, j] * diff.T) @ diff / sum_weights[j] + np.eye(features) * 1e-6

        if i % 10 == 0:
            print(f"Iteration {i} complete...")

    return means, covariances, priors
