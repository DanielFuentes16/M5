from __future__ import print_function
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import glob
import matplotlib.pyplot as plt
import numpy as np



# Device configuration
from DatasetW import DatasetW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 8
num_epochs = 150
batch_size = 32
learning_rate = 0.01
momentum = 0.0

# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=True,
#                                           transform=transforms.ToTensor(),
#                                           download=True)

#train_dataset = DatasetW(True)
#test_dataset = DatasetW(False)
#test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                          train=False,
#                                          transform=transforms.ToTensor())

# Data loader
#kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)
TRAIN_DATA_PATH = "datasets/MIT_split/train"
TEST_DATA_PATH = "datasets/MIT_split/test"

TRANSFORM_IMG = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(256),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225] )
    ])

train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#DATA_PATH = '\PycharmProjects\MNISTData'
#train_data =torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True
                                           )
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)


# functions to show an image

classes= ('coast', 'forest', 'highway', 'inside_city',
           'mountain', 'Opencountry', 'street', 'tallbuilding')
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
#dataiter = iter(train_loader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


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

        self.globalAvg = torch.nn.AvgPool2d(kernel_size=(3,3))
        self.drop_out = torch.nn.Dropout(0.2)

        self.fc1 = torch.nn.Linear(23 * 23 * 256, 128)
        self.batch = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 8)
        self.act = torch.nn.ReLU()
        self.soft = torch.nn.Softmax()

    def forward(self, x):
        """Build an actor (policy) network that maps states -> actions."""

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.globalAvg(out)
        out = out.reshape(out.size(0), -1)


        out = self.fc1(out)
        #print(out.shape)
        #out = self.drop_out(out)
        #out = self.soft(out)
        #out = self.fc2(out)
        out = self.act(out)  # [batch_size, 128]
        out = self.drop_out(out)
        out = self.fc2(out)
        #out = self.soft(out)
        print(out.shape)

        return out

model = MLP().to(device)
#print(model)
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = momentum, nesterov=False)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

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

        # Accuracy
        outputs = (outputs > 0.5).float()
        correct = (outputs == labels[:8]).float().sum()

        #if (i+1) % 100 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item(), correct/outputs.shape[0]))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:

        #images = images.reshape(-1, 28*28).to(device)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')