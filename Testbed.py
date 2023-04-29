import pickle
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from LoadCIFARFile import CIFARVectorSet


class TestBed:

    def __init__(self, batchNum: int, horizon: int):

        #self.vectors = 
        self.Streams = CIFARVectorSet(batchNum).getStreams()
        # S is a set of sets
        self.time = 0
        self.horizon = horizon


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
        if dist_i > dist_min or dist_j > dist_min:
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

    def makeCoreset(self, k):
        S_prime = []

        while(self.time < self.horizon):
            # Loop over each stream Si
            for Stream in self.Streams:
                s = Stream.getNext()

                # if size of S' < k, add s to S'
                if len(S_prime) < k:
                    if Stream.hasNext():
                        S_prime.append(s)

                else:
                    self.adjust(S_prime, s)
            self.time += 1

        return S_prime


# cifar_dict = unpickle("cifar-10-batches-py/data_batch_1")

x = TestBed(0, 1)
print(x.makeCoreset(1))

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
