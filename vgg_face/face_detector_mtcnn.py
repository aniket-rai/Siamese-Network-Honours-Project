import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mtcnn import MTCNN

from data_processing import resize

detector = MTCNN()

video_cap = cv2.VideoCapture(0)

while True:
  ret, frame = video_cap.read()
  frame = resize(frame, 400)
  anchor = (0, 0)
  width = 0
  height = 0

  try:
    region = detector.detect_faces(frame)[0]["box"]
    anchor = (region[0], region[1])
    width = region[2]
    height = region[3]
  except:
    pass

  cropped_frame = frame[anchor[1]:anchor[1]+height, anchor[0]:anchor[0]+width]
  plt.imshow(cropped_frame)
  
  cv2.rectangle(frame, anchor, (anchor[0]+width, anchor[1]+height), (0, 255, 0), 2)
  cv2.imshow('Video Stream', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_cap.release()
cv2.destroyAllWindows()