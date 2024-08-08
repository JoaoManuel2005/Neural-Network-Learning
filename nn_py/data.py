import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

def spiral(sample, clas):
    X, y = spiral_data(samples=sample, classes=clas)
    return X, y

a = 'change'