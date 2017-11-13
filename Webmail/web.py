import numpy as np 
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
# np.set_printoptions(threshold=np.inf)
def max_color(img):
	gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gimg_backup = np.copy(gimg);
	A = np.ravel(gimg)
	A = A[~(A == 255)]
	a = 15
	b = 245
	X = np.zeros(((b-a),));
	breaker = 1
	for i in range(a,b):
		count = 0
		for k in range (0,A.shape[0]):
			if i-breaker < A[k] < i+breaker:
				count += 1
		X[i-a] = count
	x = np.argmax(X) + a

	gimg[(gimg >= (x+breaker))] = 255
	gimg[(gimg <= (x-breaker))] = 255
	median = cv2.medianBlur(gimg,3)
	return median
# plt.subplot(3,3,1),plt.imshow(img,'gray')
# plt.subplot(3,3,2),plt.imshow(gimg_backup,'gray')
# plt.subplot(3,3,3),plt.imshow(gimg,'gray')
# plt.subplot(3,3,4),plt.imshow(median,'gray')
# # print(img[0:10][0:10])
# plt.show()

