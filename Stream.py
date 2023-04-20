from collections import OrderedDict



class DStream:
    def __init__(self, S: set()):

        self.values = OrderedDict()
        self.currentTime = 0

        time=0
        for s in S:
            self.values.update({time: s})
            time += 1
        
    


    
    def getNext(self):
        x= self.values[self.currentTime]
        self.currentTime +=1
        return x
    

y = {1,2,3,4,5,6,7,8,9}

x = DStream(y)

print(x.getNext())
print(x.getNext())
print(x.getNext())
print(x.getNext())
print(x.getNext())


    






