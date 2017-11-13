import numpy as np 
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
import scipy.ndimage as ndimage
import web as wb
# np.set_printoptions(threshold=np.inf)
def smooth(img):
	img = img[5:75,5:55,:]
	img= cv2.resize(img,(60,80),interpolation=cv2.INTER_CUBIC)
	img_backup = np.copy(img)
	r = img[:,:,0]
	g = img[:,:,1]
	b = img[:,:,2]
	A = np.ravel(r)
	counta = np.bincount(A)
	B = np.ravel(g)
	countb = np.bincount(B)
	C = np.ravel(b)
	countc = np.bincount(C)
	r_c = np.argmax(counta)
	g_c = np.argmax(countb)
	b_c = np.argmax(countc)
	for i in range (0,10):
		r[r == (r_c+i)] = 255
		r[r == (r_c-i)] = 255
		g[g == (g_c+i)] = 255
		g[g == (g_c-i)] = 255
		b[b == (b_c+i)] = 255
		b[b == (b_c-i)] = 255
	img_backremove = np.copy(img)
	gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gimg,254,255,cv2.THRESH_BINARY)
	thresh[thresh == 255] = 1
	thresh = np.array(thresh,dtype=np.uint8)
	kernel = np.ones((4,4), np.uint8)
	# img_er = cv2.medianBlur(thresh,3)
	img_dilation = cv2.dilate(thresh, kernel, iterations=1)
	#find all your connected components (white blobs in your image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_dilation, connectivity=8)
	#connectedComponentswithStats yields every seperated component with information on each of them, such as size
	#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
	sizes = stats[1:, -1]; nb_components = nb_components - 1

	# minimum size of particles we want to keep (number of pixels)
	#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
	min_size = 30 

	#your answer image
	img2 = np.zeros((output.shape))
	for i in range(0, nb_components):
	    if sizes[i] >= min_size:
	        img2[output == i + 1] = 255
	img2[img2 == 255] = 1
	for i in range(0,r.shape[0]):
			for j in range(0,r.shape[1]):
				if img2[i][j] == 1:
					r[i][j] = 255
					g[i][j] = 255
					b[i][j] = 255  
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 5
	ret,label,center=cv2.kmeans(Z,K,None,criteria,50,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	res2_backup = np.copy(res2)
	gimg2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
	ret,thresh2 = cv2.threshold(gimg2,240,255,cv2.THRESH_BINARY)
	thresh2[thresh2 == 255] = 1
	thresh2 = np.array(thresh2,dtype=np.uint8)
	kernel2 = np.ones((9,9), np.uint8)
	img_dilation2 = cv2.dilate(thresh2, kernel, iterations=2)
	img_erode2 = cv2.erode(img_dilation2, kernel, iterations=3)
	for i in range(0,r.shape[0]):
			for j in range(0,r.shape[1]):
				if img_erode2[i][j] == 1:
					res2[i][j][0] = 255
					res2[i][j][1] = 255
					res2[i][j][2] = 255  
	out = wb.max_color(res2)
	out_f = np.copy(res2)
	for i in range(0,r.shape[0]):
			for j in range(0,r.shape[1]):
				if out[i][j] == 255:
					out_f[i][j][0] = 255
					out_f[i][j][1] = 255
					out_f[i][j][2] = 255
	return out_f
# plt.subplot(3,3,1),plt.imshow(img,'gray')
# plt.subplot(3,3,2),plt.imshow(gimg,'gray')
# plt.subplot(3,3,3),plt.imshow(thresh,'gray')
# plt.subplot(3,3,4),plt.imshow(img2,'gray')
# plt.show()