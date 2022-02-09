
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3968, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


net = Net()