import nnfs
from nnfs.datasets import spiral_data, sine_data

nnfs.init()

def get_spiral(sample, clas):
    X, y = spiral_data(samples=sample, classes=clas)
    return X, y

def get_sine():
    X, y = sine_data()
    return X, y