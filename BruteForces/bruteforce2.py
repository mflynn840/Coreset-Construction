# Sure! Here's a better formatted list of the steps to construct a composable core set:
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

        # for each item in the steam
        for item in range(1, len(self.originalDataset)):

            eligable = True
            # Compute the distance between the new data point and each point in the current core set.
            for cItem in self.coreSet:
                if self.Idist(cItem, self.originalDataset[item]) < thresholdDistance:
                    eligable = False
                    break

            if eligable:
                self.coreSet.append(self.originalDataset[item])

    def Idist(self, Citem, newItem):
        # print("core set item is " + str(type(Citem)))
        # print("new item is " + str(type(newItem)))

        # print(Citem)
        # print(newItem)

        return self.dist(Citem[0], Citem[1], newItem[0], newItem[1])

    def dist(self, x1: float, y1: float, x2: float, y2: float):

        dist = math.sqrt((math.pow(x2 - x1, 2)) + (math.pow(y2 - y1, 2)))
        print("Distance: " + str(dist))
        return dist

    def clear(self):
        self.coreSet = []


def testCoreSetConstruction():
    # Generate synthetic dataset
    X, y = make_blobs(n_samples=100, centers=10, n_features=1, random_state=50)
    print(X)
    print(y)

    coors = []
    for i in range(0, len(X)):
        coors.append((X[i][0], y[i]))
    print("done")

    print(coors)

    coreset = Coreset(coors)

    dist = 0.0
    while dist < 1.0:
        coreset.constructBruteForce(dist)
        C = coreset.coreSet

        print("Core set is:")
        print(C)

        xs = []
        ys = []

        for c in C:
            xs.append(c[0])
            ys.append(c[1])

        print(len(xs))
        print(len(ys))

        plt.figure(figsize=(8, 6))  # Adjust the figure size to 8x6 inches
        plt.scatter(X, y, c=y)
        plt.scatter(xs, ys, marker='x', color='r', s=100)
        plt.title('Composable Coreset')
        plt.show()
        coreset.clear()
        dist += .1


testCoreSetConstruction()

#  1) Initialize an empty set of points, called the "core set".
#  2) Read in the first data point from the stream.
#  3) Add the first data point to the core set.
#  4) Read in the next data point from the stream.
#  5)
#  6) If the new data point is closer to any point in the current core set than a certain threshold distance, skip the new data point and go back to step 4.
#  7) If the new data point is not closer to any point in the current core set than the threshold distance, add the new data point to the core set.
