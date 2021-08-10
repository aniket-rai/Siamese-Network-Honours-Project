import tensorflow as tf

from data_processing import preprocess
from data_processing import generate_data
from cosine_distance import find_cosine_distance

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




