import cv2
import matplotlib.pyplot as plt

def detect_faces_cv(video_cap, faceCascade):
  cropped_frame = list()

  while len(cropped_frame) < 1:
    frame = video_cap.read();
    gray = cv2.resize(frame, (120,160), interpolation=cv2.INTER_AREA);
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # facial detection
    faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=10,
      minSize=(100,100),
      flags=cv2.CASCADE_SCALE_IMAGE
    )

    # draw bounding box
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      cropped_frame.append(frame[y:y+h, x:x+w])
  
  return cropped_frame[0], frame
