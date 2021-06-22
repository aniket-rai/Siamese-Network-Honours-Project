import os

import torch
from torch.utils.data import DataLoader

from data_processing import LFW_Train, LFW_Test

lfw_train = LFW_Train()
lfw_test = LFW_Test()

train_dataloader = DataLoader(lfw_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(lfw_test, batch_size=64, shuffle=True)

