import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
from os import listdir
from os.path import isfile, join
def segment_web1(path):
	img = cv2.imread(path)
	gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,thresh1 = cv2.threshold(gimg,127,255,cv2.THRESH_BINARY)
	thresh1[thresh1 == 255] = 1
	start = 0
	end = 0
	valid = 0
	start_bool = 0;
	break_img = [];
	break_count = 0;
	for i in range (0,thresh1.shape[1]):
		flag = 0
		for j in range (0,thresh1.shape[0]):
			if thresh1[j,i] == 0 :
				flag = 1;
				break;
		if flag == 0 and valid == 0 and start_bool:
			valid = 1
			start = i;
		if flag == 1 and valid == 0:
			start_bool = 1 
		if flag == 1 and valid == 1 and start_bool:
			valid = 0;
			end = i-1;
			mid = math.ceil((start+end)/2)
			break_img.append(mid)
			break_count = mid
			for j in range (0,thresh1.shape[0]):
				thresh1[j,mid] = 0 
		if i > (break_count + 50) and (flag == 0 or i == (thresh1.shape[1]-1)):
			if i == (thresh1.shape[1]-1):
				start = thresh1.shape[1]-1
			val = i - break_count;
			num_break = math.floor(val/29);
			temp = break_count;
			difff = math.ceil((start-end)/num_break);
			for t in range (0,num_break-1):			
				mid = temp + ((t+1)*difff);
				break_img.append(mid)
				break_count = mid
				for j in range (0,thresh1.shape[0]):
					thresh1[j,mid] = 0
		# print (break_img)
		# if len(break_img) != 5 :
		# 	print (p);
	#lab = labels[p];
	img1 = gimg[:,0:break_img[0]] 
	img1= cv2.resize(img1,(40,40),interpolation=cv2.INTER_CUBIC)
	img2 = gimg[:,break_img[0]:break_img[1]]
	img2= cv2.resize(img2,(40,40),interpolation=cv2.INTER_CUBIC)
	img3 = gimg[:,break_img[1]:break_img[2]]
	img3= cv2.resize(img3,(40,40),interpolation=cv2.INTER_CUBIC)
	img4 = gimg[:,break_img[2]:break_img[3]]
	img4= cv2.resize(img4,(40,40),interpolation=cv2.INTER_CUBIC)
	img5 = gimg[:,break_img[3]:break_img[4]]
	img5= cv2.resize(img5,(40,40),interpolation=cv2.INTER_CUBIC)
	img6 = gimg[:,break_img[4]:thresh1.shape[1]]
	img6= cv2.resize(img6,(40,40),interpolation=cv2.INTER_CUBIC)
	return img1,img2,img3,img4,img5,img6