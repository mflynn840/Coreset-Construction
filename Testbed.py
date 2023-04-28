import pickle
import numpy as np
import matplotlib.pyplot as plt


class TestBed:

    def __init__(self, fileName: str):

        self.S = set()
        #S is a set of sets
        for i in range(1, 6):
            fName = fileName + "_" + str(i)
            print(fName)
            self.S.union(self.unpickle(fName))

        

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


#cifar_dict = unpickle("cifar-10-batches-py/data_batch_1")

x = TestBed("Datasets/cifar-10-batches-py/data_batch")







#def CoresetConstruction:

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


def CoresetConstruction(Streams, k):

    S_prime = []
    sum_points = None

    # Loop over each stream Si
    for Stream in Streams:
        s = Stream.getNext()

        # if size of S' < k:
        if len(S_prime) < k:

            if Stream.hasNext():
            # add s to S'
                S_prime.append(s)

        else:
            s_min = (None, None)
            dist_min = 0

            for s_i in S_prime:
                for s_j in S_prime:
                    if s_i != s_j:
                        dist = # dist between s_i and s_j
                        if dist < dist_min:
                            dist_min = dist
                            s_min = (s_i, s_j)

            s_i = s_min[0]
            s_j = s_min[1]
            dist_i = # dist between s and s_i
            dist_j = # dist between s and s_j

            # If adding s to S' makes div(S) "better", add s to S'
            if dist_i > dist_min:
                S_prime.append(s)

                if dist_i > dist_j:
                    S_prime.remove(s_i) #?

                else:
                    S_prime.remove(s_j) #?

    return S_prime







