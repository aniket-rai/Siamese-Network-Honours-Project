from vgg_face.data_processing import preprocess
from vgg_face.cosine_distance import find_cosine_distance

def verify(img1, img2, model):
  img1 = preprocess(img1)
  img2 = preprocess(img2)

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

