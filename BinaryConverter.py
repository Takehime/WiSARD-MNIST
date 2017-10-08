import numpy as np

# Receives image as array of ints ranging from 0 to 255
def convert(image):
    average_brightness = np.mean(image)
    for i in range(0, len(image)):
        if image[i] < average_brightness:
            image[i] = 1 #preto
        else:
            image[i] = 0 #branco
    return image