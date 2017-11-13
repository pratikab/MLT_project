import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
import smooth_pratik as st
import shutil
import glob
import os
d = dict.fromkeys(string.ascii_uppercase, 0)
d1 = dict.fromkeys(string.digits, 0)
d.update(d1)
for dire in d:
	path = 'Classes/'+dire+'/*'
	r = glob.glob(path)
	for i in r:
	   os.remove(i)
	# shutil.rmtree(path) 
# np.set_printoptions(threshold=np.inf)
smooth = 1
labels = np.load('label1_5000.npy').item()
for p in range (1,4001):
	path = 'webmail_data/'+str(p)+'.jpg'
	print(p)
	img = cv2.imread(path)
	lab = labels[p];
	break_num = len(lab)
	diff = math.ceil(204/break_num)
	if break_num == 3:
		img1 = img[:,0:diff,:] 
		img1= cv2.resize(img1,(60,80),interpolation=cv2.INTER_CUBIC)
		img2 = img[:,diff+1:(2*diff),:] 
		img2= cv2.resize(img2,(60,80),interpolation=cv2.INTER_CUBIC)
		img3 = img[:,(2*diff)+1:(3*diff),:] 
		img3= cv2.resize(img3,(60,80),interpolation=cv2.INTER_CUBIC)
		path1  = 'Classes/'+lab[0]+'/'+str(d[lab[0]])+'.jpg'
		d[lab[0]] += 1
		path2  = 'Classes/'+lab[1]+'/'+str(d[lab[1]])+'.jpg'
		d[lab[1]] += 1
		path3  = 'Classes/'+lab[2]+'/'+str(d[lab[2]])+'.jpg'
		d[lab[2]] += 1
		if smooth == 1:
			imgg1 = st.smooth(img1)
			imgg2 = st.smooth(img2)
			imgg3 = st.smooth(img3)
			cv2.imwrite(path1,imgg1)
			cv2.imwrite(path2,imgg2)
			cv2.imwrite(path3,imgg3)
		else :
			cv2.imwrite(path1,img1)
			cv2.imwrite(path2,img2)
			cv2.imwrite(path3,img3)
	elif break_num == 4: 
		img1 = img[:,0:diff,:] 
		img1= cv2.resize(img1,(60,80),interpolation=cv2.INTER_CUBIC)
		img2 = img[:,diff+1:(2*diff),:] 
		img2= cv2.resize(img2,(60,80),interpolation=cv2.INTER_CUBIC)
		img3 = img[:,(2*diff)+1:(3*diff),:] 
		img3= cv2.resize(img3,(60,80),interpolation=cv2.INTER_CUBIC)
		img4 = img[:,(3*diff)+1:(4*diff),:] 
		img4= cv2.resize(img4,(60,80),interpolation=cv2.INTER_CUBIC)
		path1  = 'Classes/'+lab[0]+'/'+str(d[lab[0]])+'.jpg'
		d[lab[0]] += 1
		path2  = 'Classes/'+lab[1]+'/'+str(d[lab[1]])+'.jpg'
		d[lab[1]] += 1
		path3  = 'Classes/'+lab[2]+'/'+str(d[lab[2]])+'.jpg'
		d[lab[2]] += 1
		path4  = 'Classes/'+lab[3]+'/'+str(d[lab[3]])+'.jpg'
		d[lab[3]] += 1
		if smooth == 1:
			imgg1 = st.smooth(img1)
			imgg2 = st.smooth(img2)
			imgg3 = st.smooth(img3)
			imgg4 = st.smooth(img4)
			cv2.imwrite(path1,imgg1)
			cv2.imwrite(path2,imgg2)
			cv2.imwrite(path3,imgg3)
			cv2.imwrite(path4,imgg4)
		else :
			cv2.imwrite(path1,img1)
			cv2.imwrite(path2,img2)
			cv2.imwrite(path3,img3)
			cv2.imwrite(path4,img4)
