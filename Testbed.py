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






