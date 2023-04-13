

class LocalSearch:
    def __init__(self):
        self.coreSet = []

    def localSearch(self, S: set(), k: int):

        #Input S is a set of points
        #      k is size of core set

        #Output: S' subset of S of size k

        Sprime = set()

        Sprime = getSprime(S, k)

        
        while localSearchConditions(S, Sprime):
            Sprime.difference(Pprime.union(p))

        return Sprime
    
    def localSearchConditions(S: set(), Sprime: set()):
        #the local search conditions are as follows:

        #true if: there exists p e S\S' and p' e S' s.t 
            div(S'\{p'} U {p}) > div(S')(1+ epsilon/n)
            

    def getSPrime(self, S: set(), k: int):
        
        #return an arbitrary set of k points which contains the 2 farthest points
    
