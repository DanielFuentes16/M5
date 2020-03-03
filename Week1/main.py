from __future__ import print_function
import torch.nn
from torch.utils.data import DataLoader
from MLP import MLP
from torchvision import transforms
import torchvision
from utils import class_plot
import numpy as np
import matplotlib.pyplot as plt

debug = False

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    num_classes = 8
    num_epochs = 150
    batch_size = 32
    learning_rate = 0.01
    momentum = 0.0

    # Arrays for results
    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    test_loss = np.zeros(num_epochs)

    # Datasets
    TRAIN_DATA_PATH = "./datasets/MIT_split/train"
    TEST_DATA_PATH = "./datasets/MIT_split/test"

    TRANSFORM_IMG = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=True)
    if debug:
        class_plot(train_data)

    model = MLP().to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=False)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy
            _, outputs = torch.max(outputs.data, 1)
            correct += (outputs == labels).sum().item()

        accuracy = correct / len(train_data) * 100
        train_acc[epoch] = accuracy
        train_loss[epoch] = loss.item()
        print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                          loss.item(), accuracy), flush=True)

        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / len(test_data) * 100
            test_acc[epoch] = accuracy
            test_loss[epoch] = loss.item()
            print('Test: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                                            total_step,
                                                                                            loss.item(),
                                                                                            accuracy), flush=True)
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    # summarize history for accuracy
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss.jpg')

if __name__ == "__main__":
    main()
