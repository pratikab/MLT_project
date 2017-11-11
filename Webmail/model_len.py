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
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16,20,5)
		# self.pool = nn.MaxPool2d(2, 2)
		#self.fc1 = nn.Linear(20 * 22 * 7, 120)
		self.fc1 = nn.Linear(20 * 7 * 22, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84,2)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		# x = F.relu(self.conv1(x))
		# x = F.relu(self.conv2(x))
		x = x.view(-1, 20 * 7 * 22)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
