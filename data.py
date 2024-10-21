import nnfs
from nnfs.datasets import spiral_data, sine_data
from zipfile import ZipFile
import os, urllib, urllib.request
import numpy as np
import matplotlib.pyplot as plt
import cv2

nnfs.init()

def get_spiral(sample, clas):
    X, y = spiral_data(samples=sample, classes=clas)
    return X, y

def get_sine():
    X, y = sine_data()
    return X, y

def get_MNIST():
    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}')
        urllib.request.urlretrieve(URL, FILE)

    print('Unzipping images...')
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)
    
    print('Done!')

def load_MNIST_image_data(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        if label == '.DS_Store':
            continue
        for file in os.listdir(os.path.join(path, dataset, label)):
            if file == '.DS_Store':
                continue
            image = cv2.imread(os.path.join(path, dataset, label, file),
                            cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    X, y = load_MNIST_image_data('train', path)
    X_test, y_test = load_MNIST_image_data('test', path)

    return X, y, X_test, y_test


