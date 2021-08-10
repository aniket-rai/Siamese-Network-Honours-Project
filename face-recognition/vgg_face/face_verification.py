import os
import datetime

from vgg_face.data_processing import preprocess
from vgg_face.cosine_distance import find_cosine_distance

def verify(img1, img2, model):
  img1_embedding = embedding(img1, model)
  img2_embedding = embedding(img2, model)

  distance = find_cosine_distance(img1_embedding, img2_embedding)

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

def embedding(img, model):
  return model.predict(preprocess(img))[0].tolist()

def compare(embedding_1, embeddeding_2):
  if find_cosine_distance(embedding_1, embeddeding_2) <= 0.4:
    return True
  else:
    return False

def init(model):
  img_path = "C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\images"
  files = os.listdir(img_path)
  faces = {}

  for file in files:
    timestamp = float(file[:-4])
    file = f"{img_path}/{file}"
    faces[timestamp] = embedding(file, model)

  return faces

