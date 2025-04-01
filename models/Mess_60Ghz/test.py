import math

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

import helperfunction
from dataset import WifiCSIDataset, WifiCSITestDataset
from network import NeuralNet
import matplotlib.pyplot as plt

GRID_SIZE_H = 4
GRID_SIZE_V = 5

def location_to_x_y(location):
    return location % (GRID_SIZE_H + 1), math.floor(location / GRID_SIZE_V)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = WifiCSIDataset('data/60Ghz_position_data.csv')

first_data = dataset[1]
features, labels = first_data
input_size = features.size()[0] * features.size()[1]
model = NeuralNet(input_size, 500, dataset.number_of_classes(), device)
model.load_state_dict(torch.load('output/model_CrossEntropyLoss_1.pt', weights_only=True))
model.eval()

generator1 = torch. Generator().manual_seed(42)
training_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator1)

test_loader = DataLoader(dataset=test_dataset,
                          shuffle=False)

actual_list = []
predicted_list = []

average_distance_error = 0
total_samples = 0
correct_samples = 0
location_plot = [([], []) for _ in range(20)]
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        actual_list.append(labels.item())
        predicted_list.append(predicted.item())
        location_plot[labels.item()][0].append(labels.item())
        location_plot[labels.item()][1].append(predicted.item())
        correct_samples += labels.item() == predicted.item()
        average_distance_error += torch.pairwise_distance(helperfunction.label_to_coord(labels.item()) * 0.8, helperfunction.label_to_coord(predicted.item()) * 0.8).item()
        total_samples += 1

print("avarge distance error: ", (average_distance_error/total_samples))
print("accuracy: ", (correct_samples/total_samples))
confusion_matrix = metrics.confusion_matrix(actual_list, predicted_list, normalize='true')
confusion_matrix = np.round(confusion_matrix, 2)
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size

# Plot confusion matrix with larger cells

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                            display_labels=[i for i in range(dataset.number_of_classes())])
cm_display.plot(ax=ax, cmap="Blues", values_format=".2f")  # Ensure values show correctly formatted

plt.xticks(fontsize=12)  # Adjust tick font size
plt.yticks(fontsize=12)
plt.show()


colors = plt.cm.tab20.colors[:20]
fig, ax = plt.subplots(figsize=(14, 8))
placed = {}
correct_per_possition = [0] * 20
accuracy_per_possition = [0] * 20
for i, locations in enumerate(location_plot):
    x_cor, y_cor = location_to_x_y(i)
    plt.scatter(x_cor, y_cor, color=colors[i], label=f'Location {i}')
    for location_cor, location_pred in zip(locations[0], locations[1]):
        if i != location_cor:
            assert "error"
        x_pred, y_pred = location_to_x_y(location_cor)
        x_cor, y_cor = location_to_x_y(location_pred)
        if x_cor != x_pred or y_cor != y_pred:
            print(x_pred, y_pred, x_cor, y_cor)
            plt.arrow(x_pred, y_pred, x_cor - x_pred, y_cor - y_pred, color=colors[location_pred], head_width=0.1, head_length=0.1, alpha=0.1)
        else:
            correct_per_possition[i] += 1
    accuracy_per_possition = correct_per_possition[i] / len(locations[0])
    print(f"Accuracy location {i}", accuracy_per_possition[i])
location = 0
for y in range(4):
    for x in range(5):
        plt.text(x, y, str(location), fontsize=13 * accuracy_per_possition[location], horizontalalignment='center',verticalalignment='center')
        location+=1


plt.xlabel('x location')
plt.ylabel('y location')
plt.title('Position plot on locations')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

