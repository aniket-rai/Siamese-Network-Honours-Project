import cv2

video_cap = cv2.VideoCapture(0);
faceCascade = cv2.CascadeClassifier('cascade.xml')

while(True):
  ret, frame = video_cap.read();
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # facial detection
  faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=10,
      minSize=(30,30),
      flags=cv2.CASCADE_SCALE_IMAGE
  )

  # draw bounding box
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  # Show image
  cv2.imshow('Video Stream', frame)

  # Break perma loop
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_cap.release()
cv2.destroyAllWindows()