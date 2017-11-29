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
import test_single_webfunc as st

label_test = np.load('label4001_5000.npy').item()
net_len=Net_len()

correct=0
net_len=torch.load('train_len_web.pt')
net_test=torch.load('trainmodel_web.pt')
for j in range(1000):
	path='webmail_data/'+str(4001+j)+'.jpg'
	#print(path,correct)
	label=st.test_single(path)
	if label_test[4001+j]==label:
		correct+=1
	else :
		print(path,correct,label,label_test[4001+j])

accuracy = correct/10
print(accuracy)


