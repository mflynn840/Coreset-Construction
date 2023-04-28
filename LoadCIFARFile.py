import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

#Turns cifar batches into numpy arrays
class CifarDataset:

    def __init__(self, fileName: str):
        self.file = os.path.join(os.getcwd(), "Coreset-Construction", "Datasets", "cifar-10-batches-py", fileName)
        self.makeDictionary()


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


    def showImage(self, index:int):
        # Display the image using Matplotlib
        plt.imshow(self.images[index], interpolation='bicubic')
        plt.title(f'CIFAR-10 Label: {self.labels[index]}')
        plt.show()

    def getElement(self, index: int):
        return (self.images[index], self.labels[index])

    def getDataset(self):

        dSet= set()
        for i in range(0, len(self.labels)):
            dSet.union(self.getElement(i))
        
        return dSet


#x = CifarDataset("data_batch_1")
#x.showImage(1)