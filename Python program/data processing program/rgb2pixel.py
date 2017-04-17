#convert the image in the directory into matrix of .npy format

import scipy.misc
import numpy as np
import glob
import os
import sys

inputsize = input("number of image: ")
inputsize = int(inputsize)
outputname = input("name of output file (in .npy): ")

pic_arr = np.zeros((inputsize,60,30,3))
i = 0

for infile in glob.glob("*.png"):
    file, ext = os.path.splitext(infile)
    im = scipy.misc.imread(infile, flatten=False, mode='RGB')
    pic_arr[i] = im
    i = i + 1

print(pic_arr.shape)
np.save(outputname, pic_arr)

