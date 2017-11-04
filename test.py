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




class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16 * 7 * 7, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84,36)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# x = F.relu(self.conv1(x))
		# x = F.relu(self.conv2(x))
		x = x.view(-1, 16 * 7 * 7)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x






d = dict.fromkeys(string.ascii_uppercase,[])
d1 = dict.fromkeys(string.digits,[])
d.update(d1)
cnt=0
length=0
for key in d.keys():
	mypath = 'CSE/'+key+'/';
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
		temp_path='CSE/'+key+'/'+ element
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
train = data_utils.TensorDataset(features, targets) 
train_loader = data_utils.DataLoader(train, batch_size=20, shuffle=True)



net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

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
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 10))
			running_loss = 0.0

print('Finished Training')
test_loader=train_loader
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

print('Accuracy of the network on the 6000 test images: %d %%' % (
    100 * correct / total))






	
print(inp.shape)



d = dict.fromkeys(string.ascii_uppercase,[])
d1 = dict.fromkeys(string.digits,[])
d.update(d1)
cnt=0
length=0
for key in d.keys():
	mypath = 'CSE_test/'+key+'/';
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
		temp_path='CSE_test/'+key+'/'+ element
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
outputs = net(Variable(images).float())
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = net(Variable(images).float())
    _, predicted = torch.max(outputs.data, 1)
    labels=labels.long()
    print(labels,predicted)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


