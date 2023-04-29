from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model
from numpy import array
from LoadCIFARFile import CifarDataset


x_train.shape is the shape of the input vectors

class CIFAR10Model:

    def __init__(self, fileName: str):
        self.x_train = LoadCIFARFile(fileName)
        self.makeModel()



    def makeModel(self):


        num_classes = 10

        #a Non recurrent, convolutional deep neural networks with 4 hidden layers
        model = Sequential() 
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.x_train.shape[1:], name='conv1'))
        model.add(BatchNormalization(axis=3, name='bn_conv1'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), name='conv2'))
        model.add(BatchNormalization(axis=3, name='bn_conv2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
        model.add(BatchNormalization(axis=3, name='bn_conv3'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), name='conv4'))
        model.add(BatchNormalization(axis=3, name='bn_conv4'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, name='fc1'))
        model.add(BatchNormalization(axis=1, name='bn_fc1'))
        model.add(Activation('relu'))
        model.add(Dense(num_classes, name='output'))
        model.add(BatchNormalization(axis=1, name='bn_outptut'))
        model.add(Activation('softmax'))

        self.model = model

    def saveModel(self):
        self.model.save("CIFAR10Model")

    def loadModel(self, file: str):
        self.model = load_model(file)


