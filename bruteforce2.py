#Sure! Here's a better formatted list of the steps to construct a composable core set:
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class Coreset:
    def __init__(self, originalDataset: list):
        self.originalDataset = originalDataset
        self.coreSet = []

    def constructBruteForce(self, thresholdDistance: int):

        self.coreSet.append(self.originalDataset[0])

        #for each item in the steam
        for item in range(1, len(self.originalDataset)):

            #Compute the distance between the new data point and each point in the current core set.
            for cItem in self.coreSet:
                x=1



def testCoreSetConstruction():
    # Generate synthetic dataset
    X, y = make_blobs(n_samples=100, centers=10, n_features=5, random_state=50)
    print(X)
    print(y)
    m=1
    C = np.zeros(m)

    plt.figure(figsize=(8, 6))  # Adjust the figure size to 8x6 inches
    plt.scatter(X[:, 0], X[:, 1], c=y)
    #plt.scatter(C[:, 0], C[:, 1], marker='x', color='r', s=100)
    plt.title('Composable Coreset')
    plt.show()



testCoreSetConstruction()

        

#  1) Initialize an empty set of points, called the "core set".
#  2) Read in the first data point from the stream.
#  3) Add the first data point to the core set.
#  4) Read in the next data point from the stream.
#  5)
#  6) If the new data point is closer to any point in the current core set than a certain threshold distance, skip the new data point and go back to step 4.
#  7) If the new data point is not closer to any point in the current core set than the threshold distance, add the new data point to the core set.