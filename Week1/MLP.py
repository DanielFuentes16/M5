from __future__ import print_function
import torch
import torch.nn


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256))

        self.globalAvg = torch.nn.AvgPool2d(kernel_size=(3, 3))
        self.drop_out = torch.nn.Dropout(0.2)

        self.fc1 = torch.nn.Linear(23 * 23 * 256, 128)
        self.batch = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 8)
        self.act = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.globalAvg(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.act(out)
        out = self.drop_out(out)
        out = self.fc2(out)
        out = self.soft(out)

        # [batch_size, 8]
        return out
