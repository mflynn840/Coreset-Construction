import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


cifar_dict = unpickle("cifar-10-batches-py/data_batch_1")
# Extract an image and its label from the dictionary
image = cifar_dict[b'data'][1]
label = cifar_dict[b'labels'][1]

# Reshape the image data to its original dimensions
image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))

# Display the image using Matplotlib
plt.imshow(image, interpolation='bicubic')
plt.title(f'CIFAR-10 Label: {label}')
plt.show()