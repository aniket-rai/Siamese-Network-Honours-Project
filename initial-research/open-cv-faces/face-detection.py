#!/usr/bin/env python

import cv2
import os

# These obviously change depending on where windows is on your path, but it
# should give you an idea where they will be on your system
#
# Also, lesson learned: don't assume you know where things are on the system.
# Use something like find or fd (or Get-Childitem on Powershell) to find the
# file. It took all of like a second on both systems.
#
# I don't know why I bothered finding the linux one anyway, wsl can't do cameras
# windows_cascade_name = "C:/Users/64221/anaconda3/envs/opencv-test/Library/etc/haarcascades/haarcascade_frontalface_default.xml"
windows_cascade_name = "C:/Users/64221/anaconda3/envs/opencv-test/Library/etc/haarcascades/haarcascade_smile.xml"
linux_cascade_name = "/home/erik/anaconda3/envs/opencv-test/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
cascade_name = windows_cascade_name if os.name == 'nt' else linux_cascade_name

# create a new cam object
cap = cv2.VideoCapture(0)
# initialize the face recognizer (default face haar cascade)
face_cascade = cv2.CascadeClassifier(cascade_name)
while True:
    # read the image from the cam
    _, image = cap.read()
    # converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect all the faces in the image
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
    # for every face, draw a blue rectangle
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
    cv2.imshow("image", image)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
