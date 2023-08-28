import torch
from torch import nn


class GestureNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(30, 8)
)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = x.to(self.linear_relu_stack[0].weight.dtype) 
        logits = self.linear_relu_stack(x)
        return logits


"""

self.linear_relu_stack = nn.Sequential(
            nn.Linear(60, 15),
            nn.ReLU(),
            nn.Linear(15, 8)
        )

self.linear_relu_stack = nn.Sequential(
    nn.Linear(60, 30),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(30, 8)
)

self.linear_relu_stack = nn.Sequential(
    nn.Linear(60, 30), 
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(30, 15),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(15, 8)
)

"""