#This file select n random files from one directory and move them into the other one.
#This is used for randomly selecting test set from the data
import os
import shutil
import sys
import random

i = 1
fin = input("File path to get image from: ")
fout = input("File path to put image: ")
sample = input("How many images to take: ")
sample = int(sample)

path = os.path.join(os.getcwd(),fin)
outpath =  os.path.join(os.getcwd(),fout)

files = [file for file in os.listdir(path)]
tempfiles = []

while(i<=sample):
    file = random.choice(files)
    if(file not in tempfiles):
        shutil.move(os.path.join(path, file), os.path.join(outpath, file))
        tempfiles.append(file)
        i = i+1
