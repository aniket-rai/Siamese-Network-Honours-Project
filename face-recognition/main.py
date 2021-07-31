# imports
import cv2
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np

from vgg_face.vgg_face_network import load_network
from vgg_face.face_detector_opencv import detect_faces_cv
from vgg_face.face_verification import verify

# open cv
video_cap = cv2.VideoCapture(0);
faceCascade = cv2.CascadeClassifier("/Users/aniketrai/Desktop/part-iv-project/face-recognition/vgg_face/cascade.xml")
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (0, 0, 0)

# vars
VGG_Face = load_network()
last_save = 0
last_face = np.random.default_rng(23).random((224, 224, 3))
text = ""
x = 0;

while True:
  f_start = time.time()
  face, frame, coords = detect_faces_cv(video_cap, faceCascade)
  
  if (x % 5 == 0):
    result = verify(face, last_face, VGG_Face)["verified"]

    if result:
      times = datetime.datetime.fromtimestamp(last_save)
      text = f"Last seen at {str(times)[:-10]}"
      
      if last_save > (last_save + 600):
        last_save = time.time()
        cur_time = time.localtime()
        last_face = face

        f_name = f"/Users/aniketrai/Desktop/part-iv-project/face-recognition/images/{cur_time.tm_mday}.{cur_time.tm_mon}.{cur_time.tm_year}.{cur_time.tm_hour}.{cur_time.tm_min}.{cur_time.tm_sec}.png"
        plt.imsave(f_name, face)
    else:
      text = "Unrecognised - never seen before."
      last_save = time.time()
      cur_time = time.localtime()
      last_face = face

      f_name = f"/Users/aniketrai/Desktop/part-iv-project/face-recognition/images/{cur_time.tm_mday}.{cur_time.tm_mon}.{cur_time.tm_year}.{cur_time.tm_hour}.{cur_time.tm_min}.{cur_time.tm_sec}.png"
      plt.imsave(f_name, face)
    
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = coords[0]
    text_offset_y = coords[1]

    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
  
  cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
  cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 255, 124), thickness=1)

  f_end = time.time()
  cv2.putText(frame, str(f_end - f_start)[:5], (10,30), font, font_scale, (0,0,0))
  
  cv2.imshow("Face", frame)
  x += 1

  # Break perma loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_cap.release()
cv2.destroyAllWindows()