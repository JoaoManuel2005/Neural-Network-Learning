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

def load_MNIST_image_data():
    image_data = cv2.imread('fashion_mnist_images/train/7/0002.png',
                            cv2.IMREAD_UNCHANGED)
    print(image_data)


load_MNIST_image_data()