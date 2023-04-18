import math
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class Coreset:
    def __init__(self, originalDataset: tuple):
        self.originalDataset = originalDataset
        self.coreSet = []

    def constructBruteForce(self, thresholdDistance: int):

        self.coreSet.append(self.originalDataset[0])


        # Compute the max possible dist
        max_dist = np.max(np.sqrt(np.sum(np.square(self.originalDataset[:, np.newaxis, :] - self.originalDataset), axis=2)))
        # for each item in the steam
        for item in self.originalDataset[1:]:

            eligible = True
            # Compute the distance between the new data point and each point in the current core set.
            max_item_dist = np.max(np.sqrt(np.sum(np.square(item - np.array(self.coreSet)), axis=1)))

            if max_item_dist > thresholdDistance:
                eligible = True
                for cItem in self.coreSet:
                    if self.Idist(cItem, item) < thresholdDistance:
                        eligible = False
                        break

            if eligible:
                self.coreSet.append(item)

            # When adding any more points to the core set will not affect the final result
            if max_dist - max_item_dist <= thresholdDistance:
                break

    def Idist(self, Citem, newItem):

        # Adjust the scale factor here
        return self.dist(Citem[0], Citem[1], newItem[0], newItem[1], 20)

    def dist(self, x1: float, y1: float, x2: float, y2: float, scale_factor: float):

        return math.sqrt((math.pow(x2 - x1, 2)) + (math.pow(y2 - y1, 2))) / scale_factor



def testCoreSetConstruction():
    # Generate synthetic dataset
    X, y = make_blobs(n_samples=100, centers=10, n_features=2, random_state=50)
    print(X)
    print(y)

    coreset = Coreset(X)


    coreset.constructBruteForce(.2)
    C = coreset.coreSet

    print("Core set is:")
    print(C)

    plt.figure(figsize=(8, 6))  # Adjust the figure size to 8x6 inches
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(np.array(C)[:, 0], np.array(C)[:, 1], marker='x', color='r', s=100)
    plt.title('Composable Coreset')
    plt.show()


testCoreSetConstruction()

