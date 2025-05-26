import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

from dataset import WifiCSIDataset, WifiCSITestDataset
from network import NeuralNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


#model = NeuralNet(input_size, hidden_size, num_classes).to(device)

dataset = WifiCSIDataset('data/output_5Ghz.csv', False)
training_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2, 0.0])

# get first sample and unpack
first_data = training_dataset[1]
features, labels = first_data
print("test", features, labels)
print(features.size())
input_size = features.size()[0] * features.size()[1]
print("input_size", input_size)

#old values: test dataset size: 0, train dataset size: 304, validation dataset size: 75
print(f"test dataset size: {len(test_dataset)}, train dataset size: {len(training_dataset)}, validation dataset size: {len(validation_dataset)}")
batch_size = 100
num_epochs = 100
learning_rate = 0.0001
train_loader = DataLoader(dataset=training_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False)

validation_loader = DataLoader(dataset=validation_dataset,
                          batch_size=batch_size,
                          shuffle=False)

total_samples = len(training_dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

criterion = nn.CrossEntropyLoss(size_average=False).cuda()
print(dataset.number_of_classes())
model = NeuralNet(input_size, 1000, dataset.number_of_classes(), device)

optimizer =  torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=60)
validation_losses = []
validation_accuracies = []
train_losses = []
train_accuracies = []
best_epoch = (0, None, 0)
for epoch in range(num_epochs):
    total_loss = 0
    n_correct = 0
    n_samples = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        inputs = inputs.reshape(-1, input_size).to(device)

        #noice_matrix = torch.rand(inputs.shape).to(device) * 0.0005
        #inputs = inputs + noice_matrix

        #print("inputs size", inputs.shape)
        labels = labels.to(device)
        # Forward passimport copy
        outputs = model(inputs)
        #print("output size", outputs.shape)
    
        #print("labels size", labels.shape)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)

        n_correct += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = total_loss / len(train_loader)
    train_acc = n_correct / n_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
    # Run your training process

    val_loss = 0
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in validation_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        validation_losses.append(val_loss)
        validation_accuracies.append(n_correct / n_samples)

        acc = 100.0 * n_correct / n_samples
        print(f'Epoch: {epoch + 1}/{num_epochs}, Labels {labels.shape}')
        print(f'Validation acc: {acc:2f} | Train acc:{(train_acc * 100):2f} %')

    #scheduler.step(val_loss)

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # print(range(1, epoch + 1), train_losses)
    # plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Loss Over Epochs')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1,len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    # plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Accuracy Over Epochs')
    #
    # plt.show()
    #
    # if epoch % 100 == 0 and epoch > 0:
    #     torch.save(best_epoch[1].state_dict(), f'model_CrossEntropyLoss_temp_{epoch}.pt')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
print(range(1, epoch + 1), train_losses)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1,len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.savefig('model_CrossEntropyLoss_temp.png')

exit()
print("epoch", best_epoch[2])
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = best_epoch[1](images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

torch.save(model.state_dict(), 'model_CrossEntropyLoss_1.pt')