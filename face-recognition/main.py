import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn

from data_processing import LFW_Train, LFW_Test
from train import train
from siamese_network import SiameseNetwork
from contrastive_loss import ContrastiveLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lfw_train = LFW_Train()
lfw_test = LFW_Test()
train_dataloader = DataLoader(lfw_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(lfw_test, batch_size=64, shuffle=True)

network = SiameseNetwork()
loss_function = ContrastiveLoss().to(device)
optimiser = torch.optim.RMSprop(network.parameters(), lr=1e-05, eps=1e-8, weight_decay=5e-4, momentum=0.9)
epochs = 1

train_loss, test_loss = train(network, optimiser, train_dataloader, test_dataloader, epochs, loss_function)

plt.plot(train_loss, label="Training Loss")
plt.plot(test_loss, label="Validation Loss")

