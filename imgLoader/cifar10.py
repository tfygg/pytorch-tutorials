"""
Created on Sat Apr  8 09:47:27 2017

@author: tfygg
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2 

print("\n\t\t torchvision.datasets.CIFAR10 test \n \t\t\t\t\t\t ----by tfygg")

# torchvision.datasets.CIFAR10
cifarSet = torchvision.datasets.CIFAR10(root = "../data/cifar/", train= True, download = True)
print(cifarSet[0])
img, label = cifarSet[0]
print (img)
print (label)
print (img.format, img.size, img.mode)
img.show()

