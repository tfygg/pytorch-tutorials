"""
Created on 2017.6.11

@author: tfygg
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2 

import myImageFloder as myFloder

print("\n\t\t torch.utils.data.DataLoader test \n \t\t\t\t\t\t ----by tfygg")

mytransform = transforms.Compose([
    transforms.ToTensor()
    ]
)

# torch.utils.data.DataLoader
cifarSet = torchvision.datasets.CIFAR10(root = "./data/cifar/", train= True, download = True, transform = mytransform )
cifarLoader = torch.utils.data.DataLoader(cifarSet, batch_size= 10, shuffle= False, num_workers= 2)
print len(cifarSet)
print len(cifarLoader)

for i, data in enumerate(cifarLoader, 0):
    print(data[i][0])
    # PIL
    img = transforms.ToPILImage()(data[i][0])
    img.show()
    break

# torch.utils.data.DataLoader
imgLoader = torch.utils.data.DataLoader(
         myFloder.myImageFloder(root = "./data/testImages/images", label = "./data/testImages/test_images.txt", transform = mytransform ), 
         batch_size= 2, shuffle= False, num_workers= 2)

for i, data in enumerate(imgLoader, 0):
    print(data[i][0])
    # opencv
    img2 = data[i][0].numpy()*255
    img2 = img2.astype('uint8')
    img2 = np.transpose(img2, (1,2,0))
    img2=img2[:,:,::-1]#RGB->BGR
    cv2.imshow('img2', img2)
    cv2.waitKey()
    break
