import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt
import string
from os import listdir
from os.path import isfile, join
d = dict.fromkeys(string.ascii_uppercase,[])
d1 = dict.fromkeys(string.digits,[])
d.update(d1)
for key in d.keys():
	mypath = 'Classes/'+key+'/';
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	d[key] = onlyfiles
for key in d.keys():
	lst = []
	for element in d[key]:
		lst.append('Classes/'+key+'/'+ element)
	d[key] = lst