
class PrefixAlgo:
    def __init__(self):
        self.coreSet = []

    def prefix(self, S: set(), k: int):
        #Run GMM obtaining a set Y = {y1...yk} with corresponding
            #radii r1.....rk
        
        #q <- the values from {[1, k-1]} which maximizes q * rq

        #Y[q+1] <- the prefix subsequence of Y of length q+1

        #Q[i] <- verticies of distance at most rq/2 from yi for (i = 1 -> q+1)
        
        # z <- math.floor((q+1)/2) 

        #{Qi1....Qiz} <- the z sparsest spheres

        #S' <- the centers of {Qi1,...Qiz}

        #add any set of k-z verticies from S \ Union from j=1 to z Qij to S'

        return Sprime

        

