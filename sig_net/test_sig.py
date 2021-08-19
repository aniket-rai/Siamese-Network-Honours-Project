import torch
import numpy as np

from siamese_network import SiameseNetwork
from data_processing_sig import LFW_Test
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("C:\\Users\\aniket\\Desktop\\part-iv-project\\face-recognition\\networks\\sig_net.pth", map_location=torch.device(device)), strict=False)

lfw_test = LFW_Test()
test_dataloader = DataLoader(lfw_test, batch_size=1, shuffle=True)
threshold = {}


for x in np.linspace(0.1, 2, 20):
  correct = 0
  total = 0
  for img1, img2, actualTensor in test_dataloader:
    img1 = img1.to(device)
    img2 = img2.to(device)
    actualTensor = actualTensor.to(device)

    out1, out2 = model(img1, img2)
    label = torch.pairwise_distance(out1, out2).detach().cpu().numpy()
    
    if (actualTensor.item() == 0):
      actual = "Same"
    else:
      actual = "Different"
    
    if (label > x):
      prediction = "Different"
    else:
      prediction = "Same"

    if (prediction == actual):
      correct += 1
    
    total += 1

    threshold[x] = correct/total
  
  print(f"-------------{x}------------")
  print(f"Accuracy on Test Dataset: {(correct/total)*100}%")
  print(f"Stats: Correct {correct} | Total {total}")
  print("-----------------------------")