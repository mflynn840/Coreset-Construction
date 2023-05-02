from collections import OrderedDict



class DStream:
    def __init__(self, S: set()):

        x = []
        for vector in S:
            y = []
            for value in vector:
                y.append(value)
            x.append(y)

        self.S = x
        self.values = OrderedDict()
        self.currentTime = 0

        time = 0
        for si in x:
            self.values.update({time: si})
            time += 1

    def getSet(self):
        return self.S
    
    def getSize(self):
        return len(self.S)
    
    def getNext(self):
        x= self.values[self.currentTime]
        self.currentTime +=1
        return x
    
    def hasNext(self):
        return self.currentTime<len(self.values)
    

#y = {1,2,3,4,5,6,7,8,9}

#x = DStream(y)

#while(x.hasNext()):
#    print(x.getNext())

    






