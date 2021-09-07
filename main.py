# imports
from operator import sub
import os
import cv2
import time
import datetime
from imutils import video
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

from imutils.video import VideoStream
from vgg_face.vgg_face_network import load_network_optimised
from vgg_face.face_detector_opencv import detect_faces_cv
from vgg_face.face_verification_optimised import init, embedding, compare, write_database

def nbytes(arr):
  return arr.nbytes

def face_recognition(fr_result, cropped_frame, kill_signal):
  # vars
  VGG_Face = load_network_optimised()
  last_save = 0.0
  last_face = embedding(np.zeros((224,224,3)), VGG_Face)
  recognised = False

  # initialise
  faces = init()
  faces_old = init()

  while kill_signal.empty():
    face = cropped_frame.get()

    f_rec = time.time()
    face = embedding(face, VGG_Face)
    result = compare(face, last_face)
    f_rec_c = time.time()
    print(f"Face Recognition time: {f_rec_c-f_rec}")
    
    if result:
      print("recognised from last face")
      times = datetime.datetime.fromtimestamp(last_save)
      text = f"Last seen at {str(times)[:-10]}"
      if last_save > (last_save + 300):
        # if person is the same as last time, and its been 5 mins
        # update the last "seen" date and move on
        del faces[last_save]
        last_save = time.time()
        last_face = face
        faces[last_save] = last_face
    else:
      for t_stamp, f_emb in faces.items():
        t_stamp = float(t_stamp)
        if compare(face, f_emb):
          text = f"Last seen at {str(datetime.datetime.fromtimestamp(t_stamp))[:-10]}"
          last_face = f_emb
          recognised = True
          last_save = t_stamp
          print("recognised from embeddings")

          if t_stamp > (t_stamp + 300):
            faces[time.time()] = face
            del faces[t_stamp]

          break
      
      if not(recognised):
        print("unrecognised")
        text = "Unrecognised - never seen before."
        last_save = time.time()
        last_face = face
        faces[last_save] = face

    recognised = False
    fr_result.put(text)
    if faces_old != faces:
      print("Writing faces to database now...")
      write_database(faces)
      faces_old = faces
      print("Database write complete!")


##### MAIN PROGRAM LOOP #######
if __name__ == "__main__":
  font_scale = 0.75
  font = cv2.FONT_HERSHEY_DUPLEX
  rectangle_bgr = (0, 0, 0)

  if os.name == 'posix':
    # jetson nano limitations
    device = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
    tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    # open video stream - linux
    print("Starting video stream...")
    video_cap = VideoStream(src=1).start();
    print("Video stream started")
    faceCascade = cv2.CascadeClassifier("/home/aniket/part-iv-project/face-recognition/vgg_face/cascade.xml")
  elif os.name == 'nt':
    # open video stream - windows
    vid_time = time.time()
    print("Starting video stream...")
    video_cap = VideoStream(src=0)
    video_cap.start()
    end_time = time.time()
    print(f"Video stream started - took {end_time-vid_time}s")
    faceCascade = cv2.CascadeClassifier("C:\\Users\\aniket\\Desktop\\part-iv-project\\cascade.xml")
  
  # vars
  text = "Initialising..."
  coords = [10, 30]

  # shared queues for multiprocessing
  fr_result = mp.Queue(1)
  cropped_frame = mp.Queue(1)
  kill_signal = mp.Queue()

  # sub-process
  sub_process = mp.Process(target=face_recognition, args=(fr_result, cropped_frame, kill_signal))
  sub_process.start()

  while True:
    f_start = time.time()
    
    # FACE DETECTION
    frame, faces_cropped = detect_faces_cv(video_cap.read(), faceCascade)
    faces_cropped.sort(key=nbytes, reverse=True)

    # SEND FRAME TO SUB PROCESS
    if len(faces_cropped) > 0:
      if (not(cropped_frame.empty())):
        cropped_frame.get()
      cropped_frame.put(faces_cropped[0])

    # GET TEXT FROM FACE RECOGNITION SUB PROCESS
    if not(fr_result.empty()):
      text = str(fr_result.get())

    # ADD TEXT ON FRAME TO DISPLAY
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    text_offset_x = coords[0]
    text_offset_y = coords[1]
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(frame, text, (10, 30), font, fontScale=font_scale, color=(0, 255, 124), thickness=1)

    # END TO END TIME
    f_end = time.time()
    cv2.putText(frame, str(f_end - f_start)[:5], (10,30), font, font_scale, (0,0,0))
    
    # display frame
    cv2.imshow("Face", frame)

    # Break perma loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  # cleanup program before quitting
  video_cap.stop()
  cv2.destroyAllWindows()
  print("Killing sub-process 1")
  kill_signal.put(True)
  sub_process.kill()
  sub_process.close()
  print("Sub process 1 killed. Exiting.")
