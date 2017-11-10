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




label=np.load('label_metric.npy')
mypath = 'webmail_data/';
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
length=1000
inp=np.ones((length,204,84,3))
target=np.zeros((length,))
for i in range (1,1001):
	temp_path= mypath+str(i)+'.jpg'
	temp=cv2.imread(temp_path)
	inp[i]=temp
	target[i]=len(label[i])-4


features=torch.from_numpy(inp)
targets=torch.from_numpy(target)
train = data_utils.TensorDataset(features, targets) 
train_loader = data_utils.DataLoader(train, batch_size=20, shuffle=True)
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

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
		if i % 20 == 19:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 20))
			running_loss = 0.0

print('Finished Training')
torch.save(net, 'trainmodel_web.pt')

# test_loader=train_loader
# dataiter = iter(test_loader)
# images, labels = dataiter.next()
# # images=images.double()
# # labels=labels.long()
# outputs = net(Variable(images).float())
# correct = 0
# total = 0
# for data in test_loader:
#     images, labels = data
#     outputs = net(Variable(images).float())
#     _, predicted = torch.max(outputs.data, 1)
#     labels=labels.long()
#     total += labels.size(0)
#     correct += (predicted == labels).sum()

# print('Training Accuracy: %d %%' % (
#     100 * correct / total))



# class_correct = np.zeros(36,)
# class_total = np.zeros(36,)
# for data in test_loader:
#     images, labels = data
#     outputs = net(Variable(images).float())
#     _, predicted = torch.max(outputs.data, 1)
#     #print(predicted.numpy().ravel(),labels.numpy())
#     d=predicted.numpy().ravel()
#     q=labels.numpy()
#     #print("hdvbvjs")
#     c = 1*((d==q).squeeze())
#     #print(c)
#     for i in range(0,len(c)):
#         label = int(q[i])
#         #print(label)
#         class_correct[label] += c[i]
#         class_total[label] += 1

# print(class_correct,class_total)

# for i in range(1,36):
#     print('Accuracy of %5s : %2d %%' % (
#         i, 100 * class_correct[i] / class_total[i]))


