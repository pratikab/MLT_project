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
from model import *









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
inp=np.ones((length,1,40,40))
target=np.zeros((length,))
i=0
for key in d.keys():
	lst = []
	for element in d[key]:
		temp_path='Classes_test/'+key+'/'+ element
		lst.append(temp_path)
		temp=cv2.imread(temp_path,0)
		# temp= cv2.resize(temp,(40,40),interpolation=cv2.INTER_CUBIC)
		#temp=temp.ravel()
		#print(type(temp),temp.shape,inp.shape,type(inp))
		#print(inp.shape)
		inp[i][0]=temp
		inp[i][0]=temp
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

features=torch.from_numpy(inp)
targets=torch.from_numpy(target)
#target=target.reshape((target.shape[0],))
#print(inp.shape,target.shape,type(data_utils.TensorDataset))
test = data_utils.TensorDataset(features, targets) 
test_loader = data_utils.DataLoader(test, batch_size=20, shuffle=True)


net=torch.load('trainmodel_web1.pt')

dataiter = iter(test_loader)
images, labels = dataiter.next()
# images=images.double()
# labels=labels.long()
outputs = net(Variable(images).float())
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = net(Variable(images).float())
    _, predicted = torch.max(outputs.data, 1)
    labels=labels.long()
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Training Accuracy: %d %%' % (
    100 * correct / total))
