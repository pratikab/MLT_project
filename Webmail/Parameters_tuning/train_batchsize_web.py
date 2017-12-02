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
	mypath = 'Classes_train/'+key+'/';
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
		temp_path='Classes_train/'+key+'/'+ element
		lst.append(temp_path)
		temp=cv2.imread(temp_path)
		#print(temp.shape)
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
features=torch.from_numpy(inp)
targets=torch.from_numpy(target)
#target=target.reshape((target.shape[0],))
#print(inp.shape,target.shape,type(data_utils.TensorDataset))
train = data_utils.TensorDataset(features, targets) 
for j in range(1,11):
	train_loader = data_utils.DataLoader(train, batch_size=5*j, shuffle=True)

	net = Net()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(13):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)
			#print(labels)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			
			outputs = net(inputs.float())
			loss = criterion(outputs, labels.long())
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.data[0]
			if i % 10 == 9:    # print every 2000 mini-batches
				print('[%d, %d, %5d] loss: %.3f' %
					  (j,epoch + 1, i + 1, running_loss / 10))
				running_loss = 0.0
	torch.save(net, 'train_batchsize'+str(j)+'_web.pt')



print('Finished Training')



d = dict.fromkeys(string.ascii_uppercase,[])
d1 = dict.fromkeys(string.digits,[])
d.update(d1)
cnt=0
length=0
for key in d.keys():
	mypath = 'Classes_validate/'+key+'/';
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
		temp_path='Classes_validate/'+key+'/'+ element
		lst.append(temp_path)
		temp=cv2.imread(temp_path)
		#print(temp.shape)
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
features=torch.from_numpy(inp)
targets=torch.from_numpy(target)

test = data_utils.TensorDataset(features, targets) 
test_loader = data_utils.DataLoader(test, batch_size=20, shuffle=True)



dataiter = iter(test_loader)
images, labels = dataiter.next()
# images=images.double()
# labels=labels.long()
lst = []
for j in range(1,11):
	net_test=torch.load('train_batchsize'+str(j)+'_web.pt')
	outputs = net_test(Variable(images).float())
	correct = 0
	total = 0
	for data in test_loader:
	    images, labels = data
	    outputs = net_test(Variable(images).float())
	    _, predicted = torch.max(outputs.data, 1)
	    labels=labels.long()
	    total += labels.size(0)
	    correct += (predicted == labels).sum()

	print('Training Accuracy: %d %%' % (100 * correct / total))
	lst.append((100 * correct / total))
np.save('batchsize.npy',lst)
plt.plot(lst)
plt.ylabel('Accuracy')
plt.xlabel('Batchsize')
plt.show()
