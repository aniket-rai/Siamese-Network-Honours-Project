# imports
import cv2
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np

from vgg_face.vgg_face_network import load_network
from vgg_face.face_detector_opencv import detect_faces_cv
from vgg_face.face_verification import embedding, compare

# open cv
video_cap = cv2.VideoCapture(0);
faceCascade = cv2.CascadeClassifier("/Users/aniketrai/Desktop/part-iv-project/face-recognition/vgg_face/cascade.xml")
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (0, 0, 0)

# vars
VGG_Face = load_network()
last_save = 0
last_face = embedding(np.random.default_rng(23).random((224, 224, 3)), VGG_Face)
text = ""

while True:
  f_start = time.time()
  face, frame, coords = detect_faces_cv(video_cap, faceCascade)

  face_img = face.copy()
  face = embedding(face, VGG_Face)
  result = compare(face, last_face)

  if result:
    times = datetime.datetime.fromtimestamp(last_save)
    text = f"Last seen at {str(times)[:-10]}"
    if last_save > (last_save + 300):
      # if person is the same as last time, and its been 5 mins
      # update the last "seen" date and move on
      last_save = time.time()
      last_face = face
  else:
    # if person is not the same as last person
    # TODO: compare to all known faces (in images folder)
    # else if not in any saved photos:
    text = "Unrecognised - never seen before."
    last_save = time.time()
    last_face = face

    f_name = f"/Users/aniketrai/Desktop/part-iv-project/face-recognition/images/{last_save}.png"
    plt.imsave(f_name, face_img)
  
  # Utility for adding text on image
  (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
  text_offset_x = coords[0]
  text_offset_y = coords[1]
  box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
  cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
  cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 255, 124), thickness=1)

  # end to end time
  f_end = time.time()
  cv2.putText(frame, str(f_end - f_start)[:5], (10,30), font, font_scale, (0,0,0))
  
  # display frame
  cv2.imshow("Face", frame)

  # Break perma loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_cap.release()
cv2.destroyAllWindows()