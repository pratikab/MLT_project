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
from model_len import *


net_test=Net_len()
inp=np.ones((1,3,204,84))
net_test=torch.load('train_len_web.pt')
path=str(input('please enter file name: '))
temp=cv2.imread(path)
temp=np.swapaxes(temp,0,2)
inp[0]=temp
target=np.zeros((1,))

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
    temp=np.asscalar(predicted.numpy())
    #print(temp)
    print(temp+3)
