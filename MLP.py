from __future__ import print_function
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import glob



# Device configuration
from DatasetW import DatasetW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 8
num_epochs = 1
batch_size = 100
learning_rate = 0.001
momentum = 0.9

# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=True,
#                                           transform=transforms.ToTensor(),
#                                           download=True)

#train_dataset = DatasetW(True)
test_dataset = DatasetW(False)
#test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                          train=False,
#                                          transform=transforms.ToTensor())

# Data loader
#kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)
TRAIN_DATA_PATH = "datasets/MIT_split/train"
TEST_DATA_PATH = "./images/test/"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


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

        self.fc10 = torch.nn.Linear(20, 128)
        self.fc11 = torch.nn.ReLU()
        self.fc12 = torch.nn.Softmax()
        self.out = torch.nn.Linear(128, 8)
        self.fc13 = torch.nn.BatchNorm2d(256)
        self.fc14 = torch.nn.Dropout(0.2)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        #x = state.view(state.size(0), -1)
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)

        #x = x.view(x.size(0), -1)  # [batch_size, 28*13*13=4732]
        #x = x.view(-1, 20)
        print(x.shape)
        x = F.relu(self.fc10(x))
        x = self.fc14(x)
        x = self.fc13(x)
        x = self.out(x)
        print(x.shape)
        return x

model = MLP().to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader, 0):
        # Move tensors to the configured device
        #images = images.reshape(-1, 28*28).to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')