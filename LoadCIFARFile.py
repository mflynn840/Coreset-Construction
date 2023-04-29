import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import dot
from numpy.linalg import norm
from Stream import DStream


# Turns cifar batches into numpy arrays
class CifarDataset:

    def __init__(self, batch: int):
        self.file = os.path.join(os.getcwd(), "Coreset-Construction", "Datasets", "cifar-10-batches-py", "data_batch_" + str(batch))
        self.makeDictionary()

        self.classes = ["Airplane", "Automobile",
                        "Bird", "Cat", "Deer", "Dog", "Frog",
                        "Horse", "Ship", "Truck"]

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def makeDictionary(self):
        cifar_dict = self.unpickle(self.file)
        # Extract an image and its label from the dictionary
        rawImages = cifar_dict[b'data']
        self.labels = cifar_dict[b'labels']

        self.images = []
        # Reshape the image data to its original dimensions
        for i in range(0, len(rawImages)):
            self.images.append(np.transpose(np.reshape(rawImages[i], (3, 32, 32)), (1, 2, 0)))

    def showImage(self, index: int):
        # Display the image using Matplotlib
        plt.imshow(self.images[index], interpolation='bicubic')
        plt.title("Class: " + self.classes[self.labels[index]])
        plt.show()

    def getElement(self, index: int):
        return self.images[index], self.labels[index]

    def getDataset(self):

        dSet = set()
        for i in range(0, len(self.labels)):
            dSet.union(self.getElement(i))

        return dSet


#x = CifarDataset("data_batch_1")

#for i in range(10):
#    x.showImage(i)


class CIFARVectorSet:

    def __init__(self, batch: int) ->None :

        self.file = os.path.join(os.getcwd(), "Coreset-Construction", "Datasets", "cifar10-vectors-py", "vectors-efficientnet_b" + str(batch))
        self.vectors = self.unpickle(self.file)
        #print(self.vectors[0:2])
        #print(len(self.dict))
        #print("done")

    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    #The SKLearn cosine similary is super slow!
    #def cosSim(self, index1:int, index2:int) -> float:
    #    a = self.vectors[index1]
    #    b = self.vectors[index2]

    #    return dot(a,b)/(norm(a)*norm(b))
    
    def cosSim(self, index1:int, index2:int) -> float:
        a = self.vectors[index1]
        b = self.vectors[index2]

        return dot(a,b)/(norm(a)*norm(b))

    def getStreams(self) -> set():

        #print(self.vectors.shape)
        streams = []
        for i in range(0, 5):
            #print(self.vectors[1])
            streams.append(DStream(self.vectors[i*10000: i*10000+10000]))
        
        return streams
            
            


        
x = CIFARVectorSet(0)
#y = CifarDataset()


#ones = []
#for i in range(len(y.labels)):
#    if y.labels[i] ==   
#        ones.append(i)


#print(ones)

#print(x.cosSim(1, ones[2]))

#y.showImage(1)
#y.showImage(ones[2])