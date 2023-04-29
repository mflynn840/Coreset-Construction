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
        s_min = (S_prime[0], S_prime[1])
        dist_min = self.cosine_similarty(S_prime[0], S_prime[1])

        for s_i in S_prime:
            for s_j in S_prime:

                if s_i.all() != s_j.all():

                    dist = self.cosine_similarity(s_i, s_j)
                    if dist < dist_min:
                        dist_min = dist
                        s_min = (s_i, s_j)
                    


        return s_min, dist_min


    def adjust(self, S_prime, s):

        
        s_min, dist_min = self.div(S_prime)

        
        s_i = s_min[0]
        s_j = s_min[1]

        print("S: " + str(s))
        print("SI: " +str(s_i))
        print("SJ: " + str(s_j))
        dist_i = self.cosine_similarity(s_i, s)
        dist_j = self.cosine_similarity(s_j, s)

        # If adding s to S' makes div(S) "better", add s to S'
        if dist_i > dist_min or dist_j > dist_min:
            S_prime.append(s)

            if dist_i > dist_j:
                S_prime.remove(s_i)

            else:
                S_prime.remove(s_j)


    #The SKLearn cosine similary is super slow!
    def cosine_similarity(self, a, b) -> float:

        return dot(a,b)/(norm(a)*norm(b))

    def makeCoreset(self, k):
        S_prime = []

        print("debug1")

        # Loop over each stream Si
        for Stream in self.Streams:

            if(self.time >= self.horizon):
                break
            print("time horizon not met")
            if Stream.hasNext():
                s = Stream.getNext()
            else:
                continue

            # if size of S' < k, add s to S'
            if len(S_prime) < k:
                print("Core set not yet full")
                S_prime.append(s)
            else:
                self.adjust(S_prime, s)
            self.time += 1

        return S_prime


# cifar_dict = unpickle("cifar-10-batches-py/data_batch_1")

x = TestBed(0, 6)
print(x.makeCoreset(2))

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
