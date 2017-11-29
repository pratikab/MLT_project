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
import segmentation_single_web as sc
from model import *




net_test=Net()
net_test=torch.load('trainmodel_web.pt')


d = dict.fromkeys(string.ascii_uppercase,[])
d1 = dict.fromkeys(string.digits,[])
d.update(d1)
cnt=0
length=0
for key in d.keys():
	mypath = 'Classes_test/'+key+'/';
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	length+=len(onlyfiles)
	#print(onlyfiles)
	d[key] = onlyfiles
print(length)
inp=np.ones((length,80,60,3))
target=np.zeros((length,))
i=0
for key in d.keys():
	lst = []
	for element in d[key]:
		temp_path='Classes_test/'+key+'/'+ element
		lst.append(temp_path)
		temp=cv2.imread(temp_path)
		# temp= cv2.resize(temp,(40,40),interpolation=cv2.INTER_CUBIC)
		#temp=temp.ravel()
		#print(type(temp),temp.shape,inp.shape,type(inp))
		#print(inp.shape)
		inp[i]=temp
		if (ord(key)<65):
			num=ord(key)-48
		else:
			num=ord(key)-55
		# target[i][num]=1
		target[i]=num
		i+=1
				#print(inp.shape,target.shape)
	cnt+=1
	d[key] = lst
	print(key,inp.shape)

inp=np.swapaxes(inp,1,3)
features_test=torch.from_numpy(inp)
targets_test=torch.from_numpy(target)
#target=target.reshape((target.shape[0],))
#print(inp.shape,target.shape,type(data_utils.TensorDataset))
test = data_utils.TensorDataset(features_test, targets_test) 
test_loader = data_utils.DataLoader(test, shuffle=True)



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
    correct += (predicted == labels).sum()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

