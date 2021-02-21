import cv2

img = cv2.imread('images/conda-original.png', 1)
scale = 0.6
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)

# The options for the `interpolation` variable are:
# - INTER_AREA: resampling using pixel area relation. like image zoom.
# - INTER_CUBIC: "A bicubic interpolation over 4×4 pixel neighborhood".
# - INTER_LANCOZS4: "Lanczos interpolation over 8×8 pixel neighborhood"
resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

print('Original Dimensions : ', img.shape)
print('Resized Dimensions  : ', resized.shape)

cv2.imshow('Original image', img)
cv2.imshow('Resized image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
