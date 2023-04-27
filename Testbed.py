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
def CoresetConstruction(S, k):
    S_prime = []
    sum_points = None

    # Loop over each stream Si
    for Si in S:
        s = Si.getNext()

        # if size of S' < k:
        if len(S_prime) < k:
            # add s to S'
            S_prime.append(s)
            if sum_points is None:
                sum_points = s
            else:
                sum_points += s

        else:
            avg_points = sum_points / len(S_prime)

            deviation = s - avg_points

            min_dist = min([abs(s - p) for p in S_prime])

            w = deviation / min_dist

            w_prime = [deviation / min_dist for p in S_prime]

            sum_w_prime = sum(w_prime)

            # If adding s to S' makes div(S) "better", add s to S'
            if w > sum_w_prime / len(S_prime):
                S_prime.append(s)
                sum_points += s
                sum_points -= S_prime.pop(0)

    return S_prime







