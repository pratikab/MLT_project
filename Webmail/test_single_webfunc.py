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
from model import *
import segmentation_single_web as sc
def test_single(path):
	net_len=Net_len()
	inp=np.ones((1,3,204,84))
	net_len=torch.load('train_len_web.pt')
	# path=str(input('please enter file name: '))
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
	outputs = net_len(Variable(images).float())
	correct = 0
	total = 0
	for data in test_loader:
		images, labels = data
		outputs = net_len(Variable(images).float())
		_, predicted = torch.max(outputs.data, 1)
		labels=labels.long()
		#print(labels,predicted)
		total += labels.size(0)
		length=np.asscalar(predicted.numpy())+3
		#print(temp)
		#print(temp+3)


	net_test=Net()
	net_test=torch.load('trainparams13_web.pt')

	inp=np.ones((length,80,60,3))
	target=np.zeros((length,))
	if(length==3):
		img1,img2,img3=sc.segment_web(path,length)
		inp[0]=img1
		inp[1]=img2
		inp[2]=img3
	else:
		img1,img2,img3,img4=sc.segment_web(path,length)
		inp[0]=img1
		inp[1]=img2
		inp[2]=img3
		inp[3]=img4

	inp=np.swapaxes(inp,1,3)
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
	lst=[]
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
			lst.append(chr(temp+48))
		else:
			lst.append(chr(temp+55))
	lst=''.join(lst)
	return lst
	