#!/usr/bin/env python

import cv2

# There are a couple of options for the second argument of imread:
# 0 : the default - any depth
# 1 : colour - note that channels are in BGR order (for some reason)
# 2 : grayscale
# img is a matrix
img = cv2.imread('images/conda-original.png', 2)

# The title of the window you open, and the image variable. Doesn't work on WSL
cv2.imshow('image', img)

# So that the image doesn't close immediately
cv2.waitKey()

cv2.destroyAllWindows()

# Returns the status of whether it did write to the file
status = cv2.imwrite('images/conda-new.png', img)
print('Image written successfully: ', status)
