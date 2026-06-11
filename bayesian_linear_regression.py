import numpy as np
import matplotlib.pyplot as plt

def data_generator(n =20):
    X = np.random.uniform(-1, 1, (n,1))
    noise = np.random.normal(0, 0.1, (n,1))
    y = -0.3 * X + 0.5 + noise

    X_design = np.hstack([np.ones((n,1)), X])
    return X, y, X_design

raw_data, y, ax = data_generator() 

def linear_bayesian(X, y, alpha = 2.0, beta = 25.0):
    features = X.shape[1]
    S = alpha * np.eye(features) + beta * (X.T @ X)
    S_n = np.linalg.inv(S)

    m_n = beta * (S_n @ X.T @ y )

    return S_n, m_n

S_n, m_n = linear_bayesian(ax, y)

def prediction(S_n, m_n, alpha = 2.0, beta = 25.0):

    test_data = np.linspace(-1,1,100).reshape(-1,1)
    test_design = np.hstack([np.ones((100,1)), test_data])
    predictive_mean = test_design @ m_n

    predictive_variances = []

    for i in test_design:
        a = i @ S_n @ i
        var = (1/beta) + a

        predictive_variances.append(var)

    predictive_std = np.sqrt(predictive_variances)

    return test_data, predictive_mean, predictive_std


x_test, y_mean, y_std = prediction(S_n, m_n)

x_test = x_test.flatten()
y_mean = y_mean.flatten()

plt.figure(figsize = (8,6))

plt.scatter(raw_data, y, color = 'red', label = 'Observed Data')

plt.plot(x_test, y_mean, color = 'blue', linewidth = 2, label = 'Bayesian Mean Prediction')

plt.fill_between(x_test, 
                 y_mean - 2 * y_std, 
                 y_mean + 2 * y_std, 
                 color='blue', alpha=0.2, label='95% Confidence Interval')

plt.title("Bayesian Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()




    
