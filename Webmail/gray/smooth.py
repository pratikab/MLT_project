import numpy as np 
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
import scipy.ndimage as ndimage
# np.set_printoptions(threshold=np.inf)
def smooth(img):
	# img = cv2.imread(path)
	gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	thresh = (gimg<100)
	thresh = np.array(thresh,dtype=np.uint8)
	#find all your connected components (white blobs in your image)
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
	#connectedComponentswithStats yields every seperated component with information on each of them, such as size
	#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
	sizes = stats[1:, -1]; nb_components = nb_components - 1

	# minimum size of particles we want to keep (number of pixels)
	#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
	min_size = 30 

	#your answer image
	img2 = np.zeros((output.shape))
	#for every component in the image, you keep it only if it's above min_size
	for i in range(0, nb_components):
	    if sizes[i] >= min_size:
	        img2[output == i + 1] = 255
	#img3 = np.dstack((img2,img2,img2))
	return img2

# plt.subplot(3,3,1),plt.imshow(img,'gray')
# plt.subplot(3,3,2),plt.imshow(gimg,'gray')
# plt.subplot(3,3,3),plt.imshow(thresh,'gray')
# plt.subplot(3,3,4),plt.imshow(img2,'gray')
# plt.show()