import os
import json

from vgg_face.data_processing import preprocess
from vgg_face.cosine_distance import find_cosine_distance

def init():
  if os.name == 'posix':
    database = "/home/aniket/part-iv-project/face-recognition/faces.json"
  elif os.name == 'nt':
    database = "C:\\Users\\aniket\\Desktop\\part-iv-project\\faces.json"

  with open(database, 'r') as db_file:
    faces = json.load(db_file)
  
  return faces

def write_database(faces):
  if os.name == 'posix':
    database = "/home/aniket/part-iv-project/face-recognition/faces.json"
  elif os.name == 'nt':
    database = "C:\\Users\\aniket\\Desktop\\part-iv-project\\faces.json"

  with open(database, 'w') as db_file:
    json.dump(faces, db_file)

def embedding(img, interpreter):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  input_data = preprocess(img)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  
  output_data = interpreter.get_tensor(output_details[0]['index'])
  return output_data[0].tolist()

def verify(img1, img2, interpreter):
  out1 = embedding(img1, interpreter)
  out2 = embedding(img2, interpreter)

  distance = find_cosine_distance(out1, out2)

  if distance <= 0.4:
    verified = True
  else:
    verified = False

  verification_info = {
    "verified": verified,
    "distance": distance,
    "threshold": 0.4,
  }

  return verification_info

def compare(embedding_1, embeddeding_2):
  if find_cosine_distance(embedding_1, embeddeding_2) <= 0.4:
    return True
  else:
    return False





