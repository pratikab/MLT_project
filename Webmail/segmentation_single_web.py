import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
from os import listdir
from os.path import isfile, join
def segment_web(path,length):
	img = cv2.imread(path)
	break_num = int(length)
	diff = math.ceil(204/break_num)
	if break_num == 3:
		img1 = img[:,0:diff,:] 
		img1= cv2.resize(img1,(60,80),interpolation=cv2.INTER_CUBIC)
		img2 = img[:,diff+1:(2*diff),:] 
		img2= cv2.resize(img2,(60,80),interpolation=cv2.INTER_CUBIC)
		img3 = img[:,(2*diff)+1:(3*diff),:] 
		img3= cv2.resize(img3,(60,80),interpolation=cv2.INTER_CUBIC)
		# path  = 'Classes/'+lab[0]+'/'+str(d[lab[0]])+'.jpg'
		# d[lab[0]] += 1
		# cv2.imwrite(path,img1)
		# path  = 'Classes/'+lab[1]+'/'+str(d[lab[1]])+'.jpg'
		# d[lab[1]] += 1
		# cv2.imwrite(path,img2)
		# path  = 'Classes/'+lab[2]+'/'+str(d[lab[2]])+'.jpg'
		# d[lab[2]] += 1
		# cv2.imwrite(path,img3)
		return img1,img2,img3
	elif break_num == 4: 
		img1 = img[:,0:diff,:] 
		img1= cv2.resize(img1,(60,80),interpolation=cv2.INTER_CUBIC)
		img2 = img[:,diff+1:(2*diff),:] 
		img2= cv2.resize(img2,(60,80),interpolation=cv2.INTER_CUBIC)
		img3 = img[:,(2*diff)+1:(3*diff),:] 
		img3= cv2.resize(img3,(60,80),interpolation=cv2.INTER_CUBIC)
		img4 = img[:,(3*diff)+1:(4*diff),:] 
		img4= cv2.resize(img4,(60,80),interpolation=cv2.INTER_CUBIC)
		# path  = 'Classes/'+lab[0]+'/'+str(d[lab[0]])+'.jpg'
		# d[lab[0]] += 1
		# cv2.imwrite(path,img1)
		# path  = 'Classes/'+lab[1]+'/'+str(d[lab[1]])+'.jpg'
		# d[lab[1]] += 1
		# cv2.imwrite(path,img2)
		# path  = 'Classes/'+lab[2]+'/'+str(d[lab[2]])+'.jpg'
		# d[lab[2]] += 1
		# cv2.imwrite(path,img3)
		# path  = 'Classes/'+lab[3]+'/'+str(d[lab[3]])+'.jpg'
		# d[lab[3]] += 1
		# cv2.imwrite(path,img4)
		return img1,img2,img3,img4