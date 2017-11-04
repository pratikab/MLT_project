import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
from os import listdir
from os.path import isfile, join
d = dict.fromkeys(string.ascii_uppercase, 0)
d1 = dict.fromkeys(string.digits, 0)
d.update(d1)
# np.set_printoptions(threshold=np.inf)
# labels = np.load('label1-1001.npy').item()
mypath = '/home/pratik/csecaptcha'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
m = 0;
for file in onlyfiles:
	path = '/home/pratik/csecaptcha/'+ file
	lab = file.upper().split('.')[0]
	img = cv2.imread(path)
	print(m,lab)
	test = np.copy(img)
	img[np.where((img == [180,120,100]).all(axis = 2))] = [255,255,255]
	gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,thresh1 = cv2.threshold(gimg,200,255,cv2.THRESH_BINARY)
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
			# for j in range (0,thresh1.shape[0]):
				# thresh1[j,mid] = 0 
	img1 = gimg[:,break_img[0]-20:break_img[0]] 
	img1= cv2.resize(img1,(40,40),interpolation=cv2.INTER_CUBIC)
	img2 = gimg[:,break_img[0]:break_img[1]]
	img2= cv2.resize(img2,(40,40),interpolation=cv2.INTER_CUBIC)
	img3 = gimg[:,break_img[1]:break_img[2]]
	img3= cv2.resize(img3,(40,40),interpolation=cv2.INTER_CUBIC)
	img4 = gimg[:,break_img[2]:break_img[3]]
	img4= cv2.resize(img4,(40,40),interpolation=cv2.INTER_CUBIC)
	img5 = gimg[:,break_img[3]:break_img[3]+20]
	img5= cv2.resize(img5,(40,40),interpolation=cv2.INTER_CUBIC)
	# plt.subplot(3,3,1),plt.imshow(img,'gray')
	# plt.subplot(3,3,2),plt.imshow(gimg,'gray')
	# plt.subplot(3,3,3),plt.imshow(thresh1,'gray')
	# plt.subplot(3,3,4),plt.imshow(img1,'gray')
	# plt.subplot(3,3,5),plt.imshow(img2,'gray')
	# plt.subplot(3,3,6),plt.imshow(img3,'gray')
	# plt.subplot(3,3,7),plt.imshow(img4,'gray')
	# plt.subplot(3,3,8),plt.imshow(img5,'gray')
	# plt.subplot(3,3,9),plt.imshow(test,'gray')
	# plt.show()
	path  = '/home/pratik/MLT_PROJECT/CSE/'+lab[0]+'/'+str(d[lab[0]])+'.jpg'
	d[lab[0]] += 1
	cv2.imwrite(path,img1)
	path  = '/home/pratik/MLT_PROJECT/CSE/'+lab[1]+'/'+str(d[lab[1]])+'.jpg'
	d[lab[1]] += 1
	cv2.imwrite(path,img2)
	path  = '/home/pratik/MLT_PROJECT/CSE/'+lab[2]+'/'+str(d[lab[2]])+'.jpg'
	d[lab[2]] += 1
	cv2.imwrite(path,img3)
	path  = '/home/pratik/MLT_PROJECT/CSE/'+lab[3]+'/'+str(d[lab[3]])+'.jpg'
	d[lab[3]] += 1
	cv2.imwrite(path,img4)
	path  = '/home/pratik/MLT_PROJECT/CSE/'+lab[4]+'/'+str(d[lab[4]])+'.jpg'
	d[lab[4]] += 1
	cv2.imwrite(path,img5)
	m += 1
	if m == 20000:
		break;
# np.save('label1-1001.npy',labels)