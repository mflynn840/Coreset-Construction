import pickle
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm


class TestBed:

    def __init__(self, fileName: str):
        self.S = set()
        # S is a set of sets
        for i in range(1, 6):
            fName = fileName + "_" + str(i)
            print(fName)
            self.S.union(self.unpickle(fName))

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


    def div(self, S_prime):
        s_min = (None, None)
        dist_min = 0

        for s_i in S_prime:
            for s_j in S_prime:

                if s_i != s_j:
                    dist = self.cosine_similarity(np.reshape(s_i, (1, -1)), np.reshape(s_j, (1, -1)))
                    if dist < dist_min:
                        dist_min = dist
                        s_min = (s_i, s_j)

        return s_min, dist_min


    def adjust(self, S_prime, s):
        s_min, dist_min = self.div(S_prime)
        s_i = s_min[0]
        s_j = s_min[1]
        dist_i = self.cosine_similarity(np.reshape(s_i, (1, -1)), np.reshape(s, (1, -1)))
        dist_j = self.cosine_similarity(np.reshape(s_j, (1, -1)), np.reshape(s, (1, -1)))

        # If adding s to S' makes div(S) "better", add s to S'
        if dist_i > dist_min | dist_j > dist_min:
            S_prime.append(s)

            if dist_i > dist_j:
                S_prime.remove(s_i)

            else:
                S_prime.remove(s_j)


    #The SKLearn cosine similary is super slow!
    def cosine_similarity(self, index1:int, index2:int) -> float:
        a = self.vectors[index1]
        b = self.vectors[index2]

        return dot(a,b)/(norm(a)*norm(b))

    def CoresetConstruction(self, Streams, k):
        S_prime = []

        # Loop over each stream Si
        for Stream in Streams:
            s = Stream.getNext()

            # if size of S' < k, add s to S'
            if len(S_prime) < k:
                if Stream.hasNext():
                    S_prime.append(s)

            else:
                self.adjust(S_prime, s)

        return S_prime


# cifar_dict = unpickle("cifar-10-batches-py/data_batch_1")

x = TestBed("Datasets/cifar-10-batches-py/data_batch")

# def CoresetConstruction:

#    Input: set of streams S, k = size of coreset
#    Output: set of datapoints S'


#    loop over each stream Si:
#        grab the next point s e Si

#        if size of S' < k:
#            add s to S'

#        else:
#            if adding s to S' makes div(S) "better":
#                add s to S'


#    return S'
