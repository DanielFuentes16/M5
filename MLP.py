from __future__ import print_function
import torch
import torch.nn.functional as F
class MLP(torch.nn.Module):


    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.conv2d(3, 32, kernel_size=(3,3))
        self.fc2 = torch.nn.BatchNorm3d()
        self.fc3 = torch.nn.MaxPool2d(2)

        self.fc4 = torch.nn.Conv2d(32, 64, kernel_size=(3,3))
        self.fc5 = torch.nn.BatchNorm3d()
        self.fc6 = torch.nn.MaxPool2d(2)

        self.fc7 = torch.nn.Conv2d(64, 256, kernel_size=(1,1))
        self.fc8 = torch.nn.BatchNorm3d()

        self.fc9 = torch.nn.AvgPool2d()

        self.fc10 = torch.nn.Linear(128, 8)
        self.fc11 = torch.nn.ReLU()
        self.fc12 = torch.nn.Softmax()

        self.fc13 = torch.nn.BatchNorm3d()
        self.fc14 = torch.nn.Dropout(0.2)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        pass