import numpy as np 
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
np.set_printoptions(threshold=np.inf)
img = cv2.imread('2.jpg')
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
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gimg_backup = np.copy(gimg);
A = np.ravel(gimg)
A = A[~(A == 255)]
X = np.zeros((235,));
breaker = 9
for i in range(40,230):
	count = 0
	for k in range (0,A.shape[0]):
		if i-breaker < A[k] < i+breaker:
			count += 1
	X[i-11] = count
x = np.argmax(X) + 11

gimg[(gimg >= (x+breaker))] = 255
gimg[(gimg <= (x-breaker))] = 255
median = cv2.medianBlur(gimg,3)
plt.subplot(3,3,1),plt.imshow(img,'gray')
plt.subplot(3,3,2),plt.imshow(gimg_backup,'gray')
plt.subplot(3,3,3),plt.imshow(gimg,'gray')
plt.subplot(3,3,4),plt.imshow(median,'gray')
# print(img[0:10][0:10])
plt.show()

