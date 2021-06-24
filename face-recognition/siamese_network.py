import os

import torch
from torch import nn

class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.model = nn.Sequential(
      # input = [250,250,3]
      nn.Conv2d(3, 96, kernel_size=11), # [240, 240]
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, k=2, alpha=1e-04, beta=0.75),
      nn.MaxPool2d(2, stride=2), # [120, 120]
      nn.Conv2d(96, 256, kernel_size=5, padding=2), # [116, 116]
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
      nn.MaxPool2d(2, stride=2), # [58, 58]
      nn.Dropout2d(p=0.3),
      nn.Conv2d(256, 384, kernel_size=3, padding=1), # [56, 56]
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, padding=1), # [54, 54]
      nn.MaxPool2d(2, stride=2), # [27, 27]
      nn.Dropout2d(p=0.3),
      nn.Flatten(), # [27*27*256] mismatch; now using [30*30*256]
      nn.Linear(230400, 1024),
      nn.Dropout2d(p=0.5),
      nn.Linear(1024,128)
    )

  def forward(self, input_1, input_2):
    output_1 = self.model(input_1)
    output_2 = self.model(input_2)
  
    return output_1, output_2