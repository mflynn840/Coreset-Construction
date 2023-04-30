import pickle
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
from LoadCIFARFile import CIFARVectorSet
import pandas as pd
from pandas import DataFrame as df




class TestBed:

    def __init__(self, batchNum: int, horizon: int):

        #self.vectors = 
        self.Streams = CIFARVectorSet(batchNum).getStreams()
        # S is a set of sets
        self.time = 0
        self.horizon = horizon


    def div(self, S_prime: set()):
        #print("type of s prime[0]: " + str(type(S_prime)))
        s_min = None
        dist_min = -1

        #print("Sprime is: " + str(S_prime))
        for s_i in S_prime:
            for s_j in S_prime:

                #print("Type of SI: " + str(type(s_i)))

                if s_min == None:
                    s_min = (s_i, s_j)

                    #print("taking cosine sim of: " + str(s_i) + " and " + str(s_j))
                    dist_min = self.cosine_similarity(s_i, s_j)

                if s_i.all() != s_j.all():
                    #print("taking cosine sim of: " + str(s_i) + " and " + str(s_j))
                    dist = self.cosine_similarity(s_i, s_j)
                    if dist < dist_min:
                        dist_min = dist
                        s_min = (s_i, s_j)
                    


        return s_min, dist_min


    def adjust(self, S_prime: set(), s):

        
        s_min, dist_min = self.div(S_prime)

        
        s_i = s_min[0]
        s_j = s_min[1]

        #print("S: " + str(s))
        #print("SI: " +str(s_i))
        #print("SJ: " + str(s_j))
        dist_i = self.cosine_similarity(s_i, s)
        dist_j = self.cosine_similarity(s_j, s)

        # If adding s to S' makes div(S) "better", add s to S'

        #print("dist_i: " + str(dist_i))
        #print("dist_j: " + str(dist_j))
        #print("dist_min: " + str(dist_min))

        if not ((dist_i > dist_min) or (dist_j > dist_min)):
            print("coreset is staying the same")
        if ((dist_i > dist_min) or (dist_j > dist_min)):

            print("new element")
            #print("debug")
            S_prime.append(s)

            if dist_i > dist_j:
                S_prime.remove(s_i)
            else:

                S_prime(s_j)

    def dot(self, a, b):
        #print("a: " + str(a))
        #print("b: " + str(b))
        if(len(a) != len(b)):
            return 0
        
        sum = 0
        for i in range(len(a)):
            sum += a[i]*b[i]
        
        return sum




    #The SKLearn cosine similary is super slow!
    def cosine_similarity(self, a, b) -> float:

        #print("dot: " + str(self.dot(a,b)))
        sim = self.dot(a,b)/(norm(a)*norm(b))
        #print("sim: " + str(sim))
        return sim

    #coreset size is k*n where n is number of streams
    def makeCoreset(self, k):
        S_prime = []


        #print("debug1")

        # Loop over each stream Si

        while(True):
            for Stream in self.Streams:

                print("curr time: " + str(self.time) + "Time horizon: " + str(self.horizon) + "self.time >= self.horizon?: " + str(self.time >= self.horizon))

                #print("time horizon not met")
                if Stream.hasNext():
                    s = Stream.getNext()
                else:
                    continue

                # if size of S' < k, add s to S'
                if len(S_prime) < k:
                    #print("Core set not yet full")
                    #print("s: " + str(s))

                    S_prime.append(s)
                else:
                    self.adjust(S_prime, s)
                self.time += 1

                print("Time step: " + str(self.time) + "\nCoreset is: " + str(S_prime))
                print()
                print()

            if(self.time >= self.horizon):
                break

        return S_prime


# cifar_dict = unpickle("cifar-10-batches-py/data_batch_1")

#x = TestBed(0, 100)
#print(x.makeCoreset(9))

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



class AdjacenyCorset:

    def __init__(self, batchNum: int, horizon: int):

        #self.vectors = 
        self.Streams = CIFARVectorSet(batchNum).getStreams()
        # S is a set of sets
        self.time = 0
        self.horizon = horizon

    def makeCoreset(self, k):
        S_prime = []

        for i in range(len(self.Streams)):
            S_prime.append([])

        while(True):
            for i in range(len(self.Streams)):
                currentStream = self.Streams[i]
                #print("curr time: " + str(self.time) + "Time horizon: " + str(self.horizon) + "self.time >= self.horizon?: " + str(self.time >= self.horizon))

                #print("time horizon not met")
                if currentStream.hasNext():
                    s = currentStream.getNext()
                else:
                    continue

                # if size of S' < k, add s to S'
                if len(S_prime[i]) < k:
                    S_prime.append(s)
                else:
                    self.adjust(S_prime[i], s)
                self.time += 1

                #print("Time step: " + str(self.time) + "\nCoreset is: " + str(S_prime))
                #print()
                #print()

            if(self.time >= self.horizon):
                break

        UnionSPrimes = []
        for i in S_prime:
            for data in i:
                UnionSPrimes.append(data)

        return UnionSPrimes


    def dot(self, a, b):
        if(len(a) != len(b)):
            return 0
        sum = 0
        for i in range(len(a)):
            sum += a[i]*b[i]
        
        return sum
    
    def cosine_similarity(self, a, b) -> float:

        #print("dot: " + str(self.dot(a,b)))
        sim = self.dot(a,b)/(norm(a)*norm(b))
        #print("sim: " + str(sim))
        return sim
    

class Graph:
    def __init__(self, nodes: list()):

        #k nodes, for a lower triangular kxk matrix
        self.nodes = nodes
        self.matrix = []

        for i in range(len(nodes)-1):
            self.matrix.append([])

        print(self.matrix)

        letters = ['a','b','c','d']


        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):

                #pairs of i,j s.t i!=j and j, i is not included
                if i!=j and i>j:
                    coords = self.getIndex(i, j)

                    print(i,j)
                    #print(coords)
                    self.matrix[coords[0]].append(self.cosine_similarity(self.nodes[i], self.nodes[j]))

        print(self.matrix)



    def getIndex(self, i, j):
        return (i-1, j)


    def dot(self, a, b) -> float():
        if(len(a) != len(b)):
            return 0
        sum = 0
        for i in range(len(a)):
            sum += a[i]*b[i]
        
        return sum
    
    def cosine_similarity(self, a, b) -> float:

        #print("dot: " + str(self.dot(a,b)))
        sim = self.dot(a,b)/(norm(a)*norm(b))
        #print("sim: " + str(sim))
        return sim
    
    def printGraph(self) -> None:

        for row in self.matrix:
            for item in row:
                print(item)


nodes = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
x = Graph(nodes)
#x.printGraph()




print(x.cosine_similarity(nodes[0], nodes[1]))
print(x.cosine_similarity(nodes[2], nodes[0]))
print(x.cosine_similarity(nodes[2], nodes[1]))
print(x.cosine_similarity(nodes[3], nodes[0]))
print(x.cosine_similarity(nodes[3], nodes[1]))
print(x.cosine_similarity(nodes[3], nodes[2]))
    







    