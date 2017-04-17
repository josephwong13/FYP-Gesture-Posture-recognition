#Resize all images in the directory to 30x60 dimension

from PIL import Image
import glob
import os

size = (30,60)

for infile in glob.glob("*.png"):
    file, ext = os.path.splitext(infile)
    im = Image.open(infile)
    out = im.resize(size)
    out.save(file + "_30x60.png", "png")

