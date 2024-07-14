# For testing the input datasets

import numpy as np
from sklearn import datasets

X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )

samples , features = X.shape
print(samples)
print(features)

weights = np.zeros(features)
print(weights[1])

print(X.shape)
print(X[0:2])
print(y.shape)
print(y[0])