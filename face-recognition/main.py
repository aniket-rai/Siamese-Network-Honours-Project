# imports
import cv2
import time
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from vgg_face.vgg_face_network import load_network_optimised
from vgg_face.face_detector_opencv import detect_faces_cv
from vgg_face.face_verification_optimised import init, embedding, compare

# jetson nano limitations
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# open cv
video_cap = cv2.VideoCapture(0);
faceCascade = cv2.CascadeClassifier("C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\vgg_face\\cascade.xml")
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (0, 0, 0)

# vars
VGG_Face = load_network_optimised()
last_save = 0
last_face = embedding(np.random.default_rng(23).random((224, 224, 3)), VGG_Face)
text = ""
recognised = False

# initialise
faces = init(VGG_Face)

while True:
  f_start = time.time()
  face, frame, coords = detect_faces_cv(video_cap, faceCascade)

  face_img = face.copy()
  face = embedding(face, VGG_Face)
  result = compare(face, last_face)

  if result:
    print("recognised from last face")
    times = datetime.datetime.fromtimestamp(last_save)
    text = f"Last seen at {str(times)[:-10]}"
    if last_save > (last_save + 300):
      # if person is the same as last time, and its been 5 mins
      # update the last "seen" date and move on
      last_save = time.time()
      last_face = face
  else:
    for t_stamp, f_emb in faces.items():
      if compare(face, f_emb):
        text = f"Last seen at {str(datetime.datetime.fromtimestamp(t_stamp))[:-10]}"
        last_face = f_emb
        recognised = True
        last_save = t_stamp
        print("recognised from embeddings")
        break
    
    if not(recognised):
      print("unrecognised")
      text = "Unrecognised - never seen before."
      last_save = time.time()
      last_face = face
      f_name = f"C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\images\\{last_save}.png"
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
  recognised = False

  # Break perma loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_cap.release()
cv2.destroyAllWindows()