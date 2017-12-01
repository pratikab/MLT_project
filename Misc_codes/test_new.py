import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
from os import listdir
from os.path import isfile, join
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_cse as sc
from model import *


net_test=Net()
net_test=torch.load('mytrain_final.pt')
path=str(input('please enter file name: '))

img1,img2,img3,img4,img5=sc.segment(path)
inp=np.ones((5,1,40,40))
target=np.zeros((5,))
inp[0][0]=img1
inp[1][0]=img2
inp[2][0]=img3
inp[3][0]=img4
inp[4][0]=img5



features_test=torch.from_numpy(inp)
targets_test=torch.from_numpy(target)
#target=target.reshape((target.shape[0],))
#print(inp.shape,target.shape,type(data_utils.TensorDataset))
test = data_utils.TensorDataset(features_test, targets_test) 
test_loader = data_utils.DataLoader(test, shuffle=False)



dataiter = iter(test_loader)
images, labels = dataiter.next()
# images=images.double()
# labels=labels.long()
outputs = net_test(Variable(images).float())
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = net_test(Variable(images).float())
    _, predicted = torch.max(outputs.data, 1)
    labels=labels.long()
    #print(labels,predicted)
    total += labels.size(0)
    temp=predicted.numpy()
    #print(temp)
    if (temp<10):
    	print(np.asscalar(temp))
    else:
    	print(chr(temp+55))
    #print(predicted)
