import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
from os import listdir
from os.path import isfile, join
import torch.utils.data as data_utils 
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
d = dict.fromkeys(string.ascii_uppercase,[])
d1 = dict.fromkeys(string.digits,[])
d.update(d1)
inp=np.empty((1,1600))
cnt=0
target=np.empty((1,1))
for key in d.keys():
	mypath = 'Classes/'+key+'/';
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	#print(onlyfiles)
	d[key] = onlyfiles
for key in d.keys():
	lst = []
	for element in d[key]:
		temp_path='Classes/'+key+'/'+ element
		lst.append(temp_path)
		temp=cv2.imread(temp_path,0)
		# temp= cv2.resize(temp,(40,40),interpolation=cv2.INTER_CUBIC)
		temp=temp.ravel()
		inp=np.vstack((inp,temp))
		target=np.vstack((target,ord(key)))
		#print(inp.shape,target.shape)
	cnt+=1
	d[key] = lst
	print(key)
inp_t = torch.from_numpy(inp)
target_t = torch.from_numpy(target)
# print(inp.size(0),target.size(0))
train = data_utils.TensorDataset(inp_t, target_t) 
train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)