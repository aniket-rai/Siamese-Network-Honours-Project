import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1
        self.beta = 1
        self.margin = 1

    def forward(self, x1, x2, y):
        distance = torch.pairwise_distance(x1, x2)
        
        loss = (self.alpha * (1-y) * distance**2) + (self.beta * y * (torch.max(torch.zeros_like(distance), self.margin - distance)**2))

        return torch.mean(loss, dtype=torch.float)