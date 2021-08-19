import os

from vgg_face_network import load_network, load_network_optimised
# from face_verification import verify
from face_verification_optimised import verify
from data_processing import generate_data

# Clear console lambda
clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

# VGG_Face = load_network()
VGG_Face = load_network_optimised()
data = generate_data("all")

correct = 0
total = 0
false_pos = 0
false_neg = 0

pbar_t = len(data)
pbar = 0

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

  pbar += 1

  clearConsole()
  print(f"{pbar}/{pbar_t}")

print(f"{correct}/{total} = {100*correct/total:.2f}%")
print(f"False Positives: {false_pos} | False Negatives: {false_neg}")

# RESULTS FROM LAST RUN _ NON OPTIMISED:
# 2660/3199 = 83.15%
# FALSE POSITIVES = 452
# FALSE NEGATIVES = 87

# RESULTS FROM LATEST RUN _ OPTIMISEDL
# 3200/3200
# 2662/3199 = 83.21%
# False Positives: 451 | False Negatives: 86
