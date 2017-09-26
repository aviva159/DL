
import numpy as np 
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt 
import skimage.io 
import skimage.transform 
import tensorflow as tf 


#image process
print "Packs loaded"

#print Current Folder
cwd = os.getcwd()
print("Current folder is %s" % (cwd))

#Useful function
def print_typeshape(img):
	print("Type is %s" % (type(img)))
	print("Shape is %s" % (img.shape,))

#Load
cat = imread(cwd + "/images/cat.jpg")
print_typeshape(cat)

#Plot
plt.figure(0)
plt.imshow(cat)
plt.title("Original Image with imread")
plt.draw()
plt.show()

cat2 = imread(cwd + "/images/cat.jpg").astype(np.float)
print_typeshape(cat2)
#plot
plt.figure(0)
plt.imshow(cat2)
plt.title("Original Image with imread.astype(np.float)")
plt.draw()
plt.show()

#Load
cat3 = imread(cwd + "/images/cat.jpg").astype(np.float)
print_typeshape(cat3)
#Plot
plt.figure(0)
plt.imshow(cat3/255.)
plt.title("Original Image with imread.astype(np.float)/255.")
plt.draw()

plt.show()

#resize
catsmall = imresize(cat,[100, 100, 3])
print_typeshape(catsmall)
#plot
plt.figure(1)
plt.imshow(catsmall)
plt.title("Resized Image")
plt.draw()

plt.show()


#Grayscale
def rgb2gray(rgb):
	if len(rgb.shape) is 3:
		return np.dot(rgb[...,:3], [0.299,0.587,0.114])
	else:
		print ("Current Image if GRAY!")
		return rgb

catsmallgray = rgb2gray(catsmall)

print ("size of catsmallgray is %s" % (catsmallgray.shape,))
print ("type of catsmallgray is", type(catsmallgray))

plt.figure(2)
plt.imshow(catsmallgray, cmap = plt.get_cmap("gray"))
plt.title("[imshow] Gray Image")
plt.colorbar()
plt.draw()
plt.show()









