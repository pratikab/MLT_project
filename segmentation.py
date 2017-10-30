import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
img = cv2.imread(sys.argv[1])
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gimg,80,255,cv2.THRESH_BINARY)
thresh1[thresh1 == 255] = 1
start = -1
end = -1
valid = 0
start_bool = 0;
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
		for j in range (0,thresh1.shape[0]):
			thresh1[j,mid] = 0 

plt.subplot(2,2,1),plt.imshow(img,'gray')
plt.subplot(2,2,2),plt.imshow(gimg,'gray')
plt.subplot(2,2,3),plt.imshow(thresh1,'gray')
#plt.subplot(2,2,3),plt.imshow(edges,'gray')
plt.show()
