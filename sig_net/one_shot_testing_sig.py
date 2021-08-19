import torch
import numpy as np
import matplotlib.pyplot as plt

from siamese_network import SiameseNetwork
from data_processing_sig import LFW_Test
from torch.utils.data import DataLoader

device = 'gpu' if torch.cuda.is_available() else 'cpu'
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\networks\\sig_net.pth", map_location=torch.device(device)), strict=False)

lfw_test = LFW_Test()
test_dataloader = DataLoader(lfw_test, batch_size=1, shuffle=True)

img1, img2, actualTensor = next(iter(test_dataloader))

img1 = img1.to(device)
img2 = img2.to(device)
actualTensor = actualTensor.to(device)

out1, out2 = model(img1, img2)
label = torch.pairwise_distance(out1, out2).detach().cpu().numpy()

if (actualTensor.item() == 0):
  actual = "Same"
else:
  actual = "Different"

if (label > 1.2):
  prediction = "Different"
else:
  prediction = "Same"

_, axarr = plt.subplots(2)
plt.suptitle(f"Actual: {actual} | Prediction: {prediction}")
axarr[0].imshow(img1.cpu()[0][0])
axarr[1].imshow(img2.cpu()[0][0])
plt.show()