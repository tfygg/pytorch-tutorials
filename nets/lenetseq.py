"""
Created on 2017.6.11

@author: tfygg
"""
import torch
import torch.nn as nn

class LeNetSeq(nn.Module):
    def __init__(self):
        super(LeNetSeq, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = out.view(x.size(0), -1)
        out = self.fc(x)
        return out