import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class CoreSet:
    def __init__(self):
        self.shape = None

    def composable_coreset(self, X, w, m):
        # X: input data (n x d)
        # w: input weights (n,)
        # m: size of coreset

        n = self.shape[0]  # number of samples

        # initialize the coreset
        C = np.zeros((m, self.shape[1]))

        # compute the initial weights
        q = w / np.sum(w)

        for i in range(m):
            # compute the sub-sampling probabilities
            p = 4 * m * q / (i + 1)
            p = p / np.sum(p)  # normalize probabilities

            # select the top m points with highest probability values
            idx = np.argpartition(p, -m)[-m:]
            X_subset = X[idx, :]
            print("X_subset: " + str(X_subset))
            w_subset = w[idx]
            print("W_subset: " + str(w_subset))

            # compute the weighted mean and covariance
            mu = np.sum(X_subset * w_subset[:, np.newaxis], axis=0) / np.sum(w_subset)
            Sigma = np.cov(X_subset.T, aweights=w_subset)

            # compute the projection matrix
            V, D, _ = np.linalg.svd(Sigma)
            D_sqrt_inv = np.diag(np.sqrt(1 / D))
            P = V @ D_sqrt_inv

            # project the data onto the subspace
            X_proj = (X_subset - mu) @ P

            # compute the weights of the projected data
            w_proj = w_subset * np.sum(X_proj ** 2, axis=1)
            q = w_proj / np.sum(w_proj)

            # add the subset to the coreset
            C[i, :] = mu + (X_proj.T @ q).T @ P.T

        return C


# Generate synthetic dataset
X, y = make_blobs(n_samples=300, centers=10, n_features=5, random_state=50)

# Assign uniform weights to all samples
w = np.ones(X.shape[0])

# Create a composable coreset of size 50
m = 9
cs = CoreSet()
cs.shape = X.shape
C = cs.composable_coreset(X, w, m)
print(C)

# Plot the original dataset and the core-set
plt.figure(figsize=(8, 6))  # Adjust the figure size to 8x6 inches
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(C[:, 0], C[:, 1], marker='x', color='r', s=100)
plt.title('Composable Coreset')
plt.show()
