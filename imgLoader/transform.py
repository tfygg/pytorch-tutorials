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

print("\n\t\t torch.utils.data.DataLoader test \n \t\t\t\t\t\t ----by tfygg")

img_path = "./data/img_37.jpg"

# transforms.ToTensor()
transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

##opencv
img = cv2.imread(img_path)
print("img = ", img)
cv2.imshow("img", img)
cv2.waitKey()

img1 = transform1(img)
print("img1 = ",img1)
img_1 = img1.numpy()*255
img_1 = img_1.astype('uint8')
img_1 = np.transpose(img_1, (1,2,0))
cv2.imshow('img_1', img_1)
cv2.waitKey()

##PIL 1
img = Image.open(img_path).convert('RGB')
img2 = transform1(img)
print("img2 = ",img2)
img_2 = transforms.ToPILImage()(img2).convert('RGB')
print("img_2 = ",img_2)
img_2.show()

# transforms.Normalize()
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))#range [0, 255] ->[-1.0,1.0]
    ]
)
transform3 = transforms.Compose([
    transforms.Normalize(mean = (1, 1, 1), std = (2, 2, 2))#range [-1.0,1.0] -> [0.0,1.0]
    ]
)

img = Image.open(img_path).convert('RGB')
img3 = transform2(img)
print("img3 = ",img3)
img_3 = transforms.ToPILImage()(transform3(img3)).convert('RGB')
print("img_3 = ",img_3)
img_3.show()


# transforms.CenterCrop()
transform3 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.ToPILImage(),
    transforms.CenterCrop((300, 300)),
    ]
)

img = Image.open(img_path).convert('RGB')
img3 = transform3(img)
img3.show()

# transforms.RandomCrop()
transform4 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.ToPILImage(),
    transforms.RandomCrop((300,300)),
    ]
)

img = Image.open(img_path).convert('RGB')
img3 = transform4(img)
img3.show()

