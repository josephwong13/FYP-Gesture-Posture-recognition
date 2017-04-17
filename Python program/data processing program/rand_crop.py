#This file randomly crop (30,60) images from all images in the directory, and save the cropped image to the same folder.

from PIL import Image
import glob
import os
from random import randint

numberOfCrop = input("number of window to crop from a image: ")
numberOfCrop = int(numberOfCrop)

for infile in glob.glob("*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    width,height = im.size

    width = width - 30
    height = height - 60
    i = 1

    while (i<=numberOfCrop):
        left = randint(0,width)
        upper = randint(0,height)
        right = left + 30
        lower = upper + 60
        box = (left, upper, right, lower)
        #randomize an area to crop
        im2 = im.crop(box)
        numberOfFile = str(i)
        im2.save(file + "_rnd_window_" + numberOfFile + ".png" , "png")
        i = i + 1

        