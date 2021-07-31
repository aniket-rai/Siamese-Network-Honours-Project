from vgg_face_network import load_network
from face_verification import verify
from data_processing import generate_data

VGG_Face = load_network()
data = generate_data("all")

correct = 0
total = 0
false_pos = 0
false_neg = 0

for img1, img2, label in data:
  try:
    if(verify(img1, img2, VGG_Face)["verified"] == label):
      correct += 1
    elif (label == False):
      false_pos += 1
    else:
      false_neg += 1
    total += 1
  except:
    print(img1, img2)

print(f"{correct}/{total} = {100*correct/total:.2f}%")
print(f"False Positives: {false_pos} | False Negatives: {false_neg}")

# RESULTS FROM LAST RUN:
# 2660/3199 = 83.15%
# FALSE POSITIVES = 452
# FALSE NEGATIVES = 87