#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

# images/p2ecse.jpg
img = cv2.imread('images/conda-original.png', 1)

# We can subscript the image, seeing as it's a numpy ndarray
# print(img[100, 100])

# We can also find out about the image's shape
[height, width, channels] = img.shape
size = img.size
# print('Image Height       : ' , height)
# print('Image Width        : ' , width)
# print('Number of Channels : ' , channels)
# print('Image Size         : ', size)

# We can split the image into its constituent channels, and merge them back
# b, g, r = cv2.split(img)
# img = cv2.merge((b, g, r))

# Although that is quite slow, and numpy index slicing is generally better
b = img[:,:,0]

# We can convert the image to another format e.g. to cv2.COLOR_BGR2GRAY
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# We can add a border to an image (e.g. photo frames, correct dimensions to fit
# a neural network's expected input size).
BLUE = [0, 0, 255]
replicate  = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
reflect    = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
wrap       = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_WRAP)
constant   = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)

plt.subplot(231), plt.imshow(img,        'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate,  'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect,    'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap,       'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant,   'gray'), plt.title('CONSTANT')
plt.show()
