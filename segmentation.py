import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
d = dict.fromkeys(string.ascii_uppercase, 0)
d1 = dict.fromkeys(string.digits, 0)
d.update(d1)
# np.set_printoptions(threshold=np.inf)
labels = np.load('label1-1001.npy').item()
for p in range (1, 1001):
	path = '/home/pratik/ML_DATASET/'+str(p)+'.jpg'
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
	lab = labels[p];
	img1 = gimg[:,0:break_img[0]] 
	img2 = gimg[:,break_img[0]:break_img[1]]
	img3 = gimg[:,break_img[1]:break_img[2]]
	img4 = gimg[:,break_img[2]:break_img[3]]
	img5 = gimg[:,break_img[3]:break_img[4]]
	img6 = gimg[:,break_img[4]:thresh1.shape[1]]
	path  = '/home/pratik/MLT_PROJECT/Classes/'+lab[0]+'/'+str(d[lab[0]])+'.jpg'
	d[lab[0]] += 1
	cv2.imwrite(path,img1)
	path  = '/home/pratik/MLT_PROJECT/Classes/'+lab[1]+'/'+str(d[lab[1]])+'.jpg'
	d[lab[1]] += 1
	cv2.imwrite(path,img2)
	path  = '/home/pratik/MLT_PROJECT/Classes/'+lab[2]+'/'+str(d[lab[2]])+'.jpg'
	d[lab[2]] += 1
	cv2.imwrite(path,img3)
	path  = '/home/pratik/MLT_PROJECT/Classes/'+lab[3]+'/'+str(d[lab[3]])+'.jpg'
	d[lab[3]] += 1
	cv2.imwrite(path,img4)
	path  = '/home/pratik/MLT_PROJECT/Classes/'+lab[4]+'/'+str(d[lab[4]])+'.jpg'
	d[lab[4]] += 1
	cv2.imwrite(path,img5)
	path  = '/home/pratik/MLT_PROJECT/Classes/'+lab[5]+'/'+str(d[lab[5]])+'.jpg'
	d[lab[5]] += 1
	cv2.imwrite(path,img6)
# np.save('label1-1001.npy',labels)
# plt.subplot(3,3,1),plt.imshow(img,'gray')
# plt.subplot(3,3,2),plt.imshow(gimg,'gray')
# plt.subplot(3,3,3),plt.imshow(thresh1,'gray')
# plt.subplot(3,3,4),plt.imshow(img1,'gray')
# plt.subplot(3,3,5),plt.imshow(img2,'gray')
# plt.subplot(3,3,6),plt.imshow(img3,'gray')
# plt.subplot(3,3,7),plt.imshow(img4,'gray')
# plt.subplot(3,3,8),plt.imshow(img5,'gray')
# plt.subplot(3,3,9),plt.imshow(img6,'gray')
# #plt.subplot(2,2,3),plt.imshow(edges,'gray')
# plt.show()
