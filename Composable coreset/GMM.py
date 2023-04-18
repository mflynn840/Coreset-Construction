
#def GMM():

    #Input S: a set of points, k: size of the subset
    #Output S' a subset of S of size k.

    #S' ← An arbitrary point a
    #for i = 2 -> k:
        #p <- p ∈ S \ S' which maximizes min(x∈S')(dist(p, x))
        #S' <- S' union {p}


    #return S


from math import dist
import random


class GMMCoresetConstruction:

    def __init__(self):
        self.coreset = []

    def GMM(self, S: set(), k: int):

    #Input S: a set of points, k: size of the subset
    #Output S' a subset of S of size k.
        Sprime = set()
        Sprime.union(S[random.randint(0, len(S))])

        for i in range(2,k):

            newPoint = findNewPoints()#<- p ∈ S \ S' which maximizes min(x∈S')(dist(p, x))

            Sprime.union(newPoint)



        return Sprime

    def findNewPoints(self, S: set(), Sprime: set()):
        #return p ∈ S \ S' which maximizes min(x∈S')(dist(p, x))

        maxMinDistance = -1
        bestPoint = None

        for p in S.difference(Sprime):

            minDistance = -1
            point = None

            for x in Sprime:
                if dist(p, x) > minDistance:
                    minDistance = dist(p,x)
                    point = x

            if minDistance < maxMinDistance:
                maxMinDistance = minDistance
                bestPoint = point
        
        return bestPoint
        

    
    def dist(self, p, x):
        x= 1
        



    


