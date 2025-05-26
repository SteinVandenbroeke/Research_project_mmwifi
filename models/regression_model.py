import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader, Subset

from models import helperfunction
from IPython.display import Image
from tqdm import tqdm
from torchviz import make_dot

from models.networks.regression_network import Regression_neural_network

GRID_SIZE_H = 4
GRID_SIZE_V = 5

class RegressionTestModel():
    def __init__(self, name, dataset, dataset_split=[0.8, 0.1, 0.1], batch_size=50, num_epochs=500, learning_rate=0.00001,
                    Network=Regression_neural_network, hidden_size=500, random_split=True):
        self.name = f"dataset_{dataset.measurements_per_sample}__{name}_{batch_size}_{num_epochs}_{learning_rate}_{hidden_size}_{random_split}"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.Network = Network
        self.hidden_size = hidden_size
        self.generator = torch.Generator().manual_seed(0)
        self.random_split = random_split
        first_data = self.dataset[1]
        features, labels = first_data
        self.input_size = features.size()[0] * features.size()[1]

        if not os.path.isdir(f"outputs/{self.name}"):
            os.makedirs(f"outputs/{self.name}")
        self.dataset.num_classes = 2
        self.already_trained = False
        if os.path.isfile(f"outputs/{self.name}/model.pt"):
            self.model = Network(self.input_size, self.hidden_size, self.dataset.number_of_classes(), self.device)
            self.model.load_state_dict(torch.load(f"outputs/{self.name}/model.pt", weights_only=True))
            self.already_trained = True
        else:
            self.model = self.Network(self.input_size, self.hidden_size, self.dataset.number_of_classes(), self.device)

        if os.path.isfile(f"outputs/{self.name}/test_results.pt"):
            self.test_results = torch.load(f"outputs/{self.name}/test_results.pt")
        else:
            self.test_results = None

        if self.dataset.continue_sampling or not random_split:
            print("Training continous sampling")
            total_size = len(self.dataset)
            train_size = int(0.8 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size  # Ensure it sums up

            # Define fixed indices
            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, total_size))

            # Create subsets
            self.training_dataset = Subset(self.dataset, train_indices)
            self.validation_dataset = Subset(self.dataset, val_indices)
            self.test_dataset = Subset(self.dataset, test_indices)
        else:
            self.training_dataset, self.validation_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, self.dataset_split,
                                                                                           self.generator)


    def get_dataset_count(self):
        # test dataset size: 0, train dataset size: 400, validation dataset size: 100
        return len(self.test_dataset) ,  len(self.training_dataset), len(self.validation_dataset)

    def train(self, retrain=False):
        if not retrain and self.already_trained:
            print("model is already trained")
            Image(filename=f"outputs/{self.name}/train_validation.png")
            return
        self.model = self.Network(self.input_size, self.hidden_size, self.dataset.number_of_classes(), self.device)

        # test dataset size: 0, train dataset size: 400, validation dataset size: 100
        print(f"test dataset size: {len(self.test_dataset)}, train dataset size: {len(self.training_dataset)}, validation dataset size: {len(self.validation_dataset)}")

        train_loader = DataLoader(dataset=self.training_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True)

        validation_loader = DataLoader(dataset=self.validation_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False)


        total_samples = len(self.training_dataset)
        n_iterations = math.ceil(total_samples / 4)
        print(total_samples, n_iterations)

        criterion = nn.MSELoss().cuda()
        print(self.dataset.number_of_classes())

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        validation_losses = []
        validation_accuracies = []
        train_losses = []
        train_accuracies = []
        best_epoch = (0, None)
        counter = {}
        for epoch in tqdm(range(self.num_epochs)):
            total_loss = 0
            n_correct = 0
            n_samples = 0
            for i, (inputs, labels) in enumerate(train_loader):
                for label in labels:
                    if label.item() not in counter.keys():
                        counter[label.item()] = (0, 0)
                    counter[label.item()] = (counter[label.item()][0] + 1, counter[label.item()][1])

                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                inputs = inputs.reshape(-1, self.input_size).to(self.device)
                # print("inputs", inputs)
                # noice_matrix = torch.rand(inputs.shape).to(device) * 0.0005
                # inputs = inputs + noice_matrix

                # print("inputs size", inputs.shape)
                target_coords = torch.stack([self.label_to_coord(lbl) for lbl in labels]).to(self.device)
                # Forward passimport copy
                outputs = self.model(inputs)
                # print("output size", outputs.shape)

                # print("labels size", labels.shape)
                loss = criterion(outputs, target_coords)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                correct_compare = target_coords.eq(torch.round(outputs))
                count = torch.sum(torch.all(correct_compare == True, dim=1))

                n_samples += labels.size(0)
                n_correct += count.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = total_loss / len(train_loader)
            train_acc = n_correct / n_samples

            if train_acc > best_epoch[0]:
                best_epoch = (train_acc, self.model)

            train_losses.append(train_loss)
            scheduler.step(total_loss)
            train_accuracies.append(train_acc)

            # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
            # Run your training process

            val_loss = 0
            with torch.no_grad():
                n_correct = 0
                n_samples = 0
                for inputs, labels in validation_loader:
                    inputs = inputs.reshape(-1, self.input_size).to(self.device)
                    labels = labels.to(self.device)

                    target_coords = torch.stack([self.label_to_coord(lbl) for lbl in labels]).to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, target_coords)
                    val_loss += loss.item()
                    # max returns (value ,index)
                    # print("labels", target_coords.shape)
                    # print("outputs", torch.round(outputs).shape)
                    # print("compare", target_coords.eq(torch.round(outputs)))
                    correct_compare = target_coords.eq(torch.round(outputs))
                    count = torch.sum(torch.all(correct_compare == True, dim=1))
                    # print(count.item())
                    #
                    # _, predicted = torch.max(outputs.data, 1)
                    n_samples += labels.size(0)
                    #
                    n_correct += count.item()

                validation_losses.append(val_loss)
                validation_accuracies.append(n_correct / n_samples)

                acc = 100.0 * n_correct / n_samples

                # print(f'Epoch: {epoch + 1}/{self.num_epochs}, Labels {labels.shape}')
                # print(f'Validation acc: {acc:2f} | Train acc:{(train_acc * 100):2f} %')

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Over Epochs')

            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy Over Epochs')
            plt.show()

        print(counter)

        plt.savefig(f"outputs/{self.name}/train_validation.png")
        #Image(filename=f"outputs/{self.name}/train_validation.png")

        self.model = best_epoch[1]
        torch.save(self.model.state_dict(), f"outputs/{self.name}/model.pt")

    def location_to_x_y(self, location):
        return location % (GRID_SIZE_H + 1), math.floor(location / GRID_SIZE_V)

    def coord_to_label(self, coord):
        x = coord[0][0].item()
        y = coord[0][1].item()
        assert x <= GRID_SIZE_V and y <= GRID_SIZE_H, f"error {x}-{y}"
        label = x + (y * GRID_SIZE_V)
        assert label <= 20
        return label

    def label_to_coord(self, label):
        return torch.tensor(list(self.location_to_x_y(label)), dtype=torch.float32)

    def test(self, retest=False):
        if not retest and self.test_results is not None:
            print("--Test already ran--")
            print("avarge distance error: ", self.test_results["avg_distance_error"])
            print("accuracy: ", self.test_results["accuracy"])
            Image(filename=f"outputs/{self.name}/confusion_matrix.png")
            print(f"outputs/{self.name}/confusion_matrix.png")
            return

        print("--Run test--")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.eval()

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False)

        actual_list = []
        predicted_list = []

        average_distance_error = 0
        total_samples = 0
        correct_samples = 0
        location_plot = [([], []) for _ in range(20)]
        accuracy_per_possition = [0] * 20
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.reshape(-1, self.input_size).to(self.device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                target_coords = torch.stack([self.label_to_coord(lbl) for lbl in labels]).to(device)
                # max returns (value ,index)
                average_distance_error += torch.pairwise_distance(target_coords * 0.5, outputs * 0.5).item()
                total_samples += 1;
                predicted = int(self.coord_to_label(torch.abs(torch.round(outputs))))
                actual_list.append(labels.item())
                correct_samples += labels.item() == predicted
                predicted_list.append(predicted)
                location_plot[labels.item()][0].append(outputs[0][0].item())
                location_plot[labels.item()][1].append(outputs[0][1].item())
                total_samples += 1

        print("avarge distance error: ", (average_distance_error / total_samples))
        print("accuracy: ", (correct_samples / total_samples))

        self.test_results = {
            "avg_distance_error": (average_distance_error / total_samples),
            "accuracy": (correct_samples / total_samples)
        }
        torch.save(self.test_results, f"outputs/{self.name}/test_results.pt")

        confusion_matrix = metrics.confusion_matrix(actual_list, predicted_list, normalize='true')
        confusion_matrix = np.round(confusion_matrix, 2)
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size

        # Plot confusion matrix with larger cells
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                    display_labels=[i for i in range(20)])
        cm_display.plot(ax=ax, cmap="Blues", values_format=".2f")  # Ensure values show correctly formatted

        plt.xticks(fontsize=12)  # Adjust tick font size
        plt.yticks(fontsize=12)
        plt.show()

        plt.xticks(fontsize=12)  # Adjust tick font size
        plt.yticks(fontsize=12)
        plt.savefig(f"outputs/{self.name}/confusion_matrix.png")
        #Image(filename=f"outputs/{self.name}/confusion_matrix.png")
        print("locationplot", location_plot)
        # colors = plt.cm.tab20.colors[:20]
        # fig, ax = plt.subplots(figsize=(14, 8))
        # placed = {}
        # correct_per_possition = [0] * 20
        # for i, locations in enumerate(location_plot):
        #     x_cor, y_cor = self.location_to_x_y(i)
        #     plt.scatter(x_cor, y_cor, color=colors[i], label=f'Location {i}')
        #     for location_cor, location_pred in zip(locations[0], locations[1]):
        #         if i != location_cor:
        #             assert "error"
        #         x_pred, y_pred = self.location_to_x_y(location_cor)
        #         x_cor, y_cor = self.location_to_x_y(location_pred)
        #         if x_cor != x_pred or y_cor != y_pred:
        #             print(location_pred)
        #             plt.arrow(x_pred, y_pred, x_cor - x_pred, y_cor - y_pred, color=colors[location_pred],
        #                       head_width=0.1, head_length=0.1, alpha=0.1)
        #         else:
        #             correct_per_possition[i] += 1
        #     accuracy_per_possition[i] = correct_per_possition[i] / len(locations[0])

        location = 0
        for y in range(4):
            for x in range(5):
                plt.text(x, y, str(location), fontsize=11 * (accuracy_per_possition[location] + 0.5),
                         horizontalalignment='center', verticalalignment='center')
                location += 1

        plt.xlabel('x location')
        plt.ylabel('y location')
        plt.title('Position plot on locations')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"outputs/{self.name}/testing_locations.png")
        #Image(filename=f"outputs/{self.name}/testing_locations.png")

        colors = plt.cm.tab20.colors[:20]
        fig, ax = plt.subplots(figsize=(14, 8))
        print("locationplot", location_plot)
        for i, location in enumerate(location_plot):
            plt.scatter(location[0], location[1], color=colors[i], label=f'Location {i}')

        location = 0
        for y in range(4):
            for x in range(5):
                plt.text(x, y, str(location), fontsize=12, horizontalalignment='center', verticalalignment='center')
                location += 1

        plt.xlabel('x location')
        plt.ylabel('y location')
        plt.title('Position plot on locations')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"outputs/{self.name}/location_offsets.png")

    def model_to_onnx(self):
        dummy_input = torch.randn(1, self.input_size).to(self.device)

        self.model.eval()
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )