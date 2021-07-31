from vgg_face.data_processing import preprocess
from vgg_face.cosine_distance import find_cosine_distance

def verify(img1, img2, model):
  img1 = preprocess(img1)
  img2 = preprocess(img2)

  img1_embedding = model.predict(img1)[0].tolist()
  img2_embedding = model.predict(img2)[0].tolist()

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


