from __future__ import print_function
import torch
import torch.nn.functional as F
class MLP(torch.nn.Module):


    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = torch.nn.Conv2d(3, 32, kernel_size=(3,3))
        self.fc2 = torch.nn.BatchNorm2d(32)
        self.fc3 = torch.nn.MaxPool2d(2)

        self.fc4 = torch.nn.Conv2d(32, 64, kernel_size=(3,3))
        self.fc5 = torch.nn.BatchNorm2d(64)
        self.fc6 = torch.nn.MaxPool2d(2)

        self.fc7 = torch.nn.Conv2d(64, 256, kernel_size=(1,1))
        self.fc8 = torch.nn.BatchNorm2d(256)

        self.fc9 = torch.nn.AvgPool2d(kernel_size=(3,3))

        self.fc10 = torch.nn.Linear(1024, 128)
        self.fc11 = torch.nn.ReLU()
        self.fc12 = torch.nn.Softmax()
        self.out = torch.nn.Linear(128, 8)
        self.fc13 = torch.nn.BatchNorm1d(128)
        self.fc14 = torch.nn.Dropout(0.2)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)

        x = x.view(x.size(0), -1)  # [batch_size, 28*13*13=4732]
        print(x.shape)
        x = F.relu(self.fc10(x))
        x = self.fc14(x)
        x = self.fc13(x)
        x = self.out(x)

        return x

model = MLP()

batch_size, C, H, W = 3, 3, 32, 32
x = torch.randn(batch_size, C, H, W)
output = model(x)
print(output)