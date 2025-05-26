import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast 
from collections import defaultdict

class WifiCSIDataset(Dataset):
    def __init__(self, file_path, measurements_per_sample, recreate=False, continue_sampling=False, subtract_background_noice=False, remove_locations = [], name="", remove_names=[]):
        self.measurements_per_sample = measurements_per_sample
        self.continue_sampling = continue_sampling
        self.file_path = file_path
        self.subtract_background_noice = subtract_background_noice
        self.num_classes = 20 - len(remove_locations)
        self.remove_locations = remove_locations
        self.remove_names = remove_names
        continue_sampling_name = ""
        if continue_sampling:
            continue_sampling_name = "continue_sampling_"
        if os.path.exists(file_path.replace('.csv', f'{name}{continue_sampling_name}{self.measurements_per_sample}_{self.subtract_background_noice}_labels.pt')) and os.path.exists(
                file_path.replace('.csv', f'{name}{continue_sampling_name}{self.measurements_per_sample}_{self.subtract_background_noice}_data.pt')) and not recreate:
            self.labels = torch.load(file_path.replace('.csv', f'{name}{continue_sampling_name}{self.measurements_per_sample}_{self.subtract_background_noice}_labels.pt'))
            self.data = torch.load(file_path.replace('.csv', f'{name}{continue_sampling_name}{self.measurements_per_sample}_{self.subtract_background_noice}_data.pt'))
            print(self.data.shape)
            print(self.__getitem__(0))
        else:
            self.labels, self.data = self.__load_from_csv(file_path)
            torch.save(self.labels, file_path.replace('.csv', f'{name}{continue_sampling_name}{self.measurements_per_sample}_{self.subtract_background_noice}_labels.pt'))
            torch.save(self.data, file_path.replace('.csv', f'{name}{continue_sampling_name}{self.measurements_per_sample}_{self.subtract_background_noice}_data.pt'))
        self.n_samples = self.data.shape[0]
        print(self.__len__())

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.data[index], self.labels[index][1]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def __safe_eval(self, value):
        try:
            return ast.literal_eval(value) if isinstance(value, str) else [value]  # Convert single values into a list
        except (ValueError, SyntaxError):
            print("Error", value)
            return [value]  # Return as a single-element list if there's an issue

    def __trim_tensor(self, tensor, target_length):
        current_length, feature_dim = tensor.shape  # Get original shape

        if current_length > target_length:
            # Compute how much to trim
            excess = current_length - target_length
            trim_start = excess // 2  # Trim half from start
            trim_end = trim_start + target_length  # Compute end index
            return tensor[trim_start:trim_end, :]  # Slice tensor

        return tensor  # If already correct size, return as is

    def __split_up_evenly(self, tensor, target_length):
        num_chunks = tensor.numel() // target_length
        chunks = list(tensor[:num_chunks * target_length].split(target_length))

        # if tensor.numel() % target_length != 0:
        #     chunks.append(tensor[-target_length:])Leakage

        return chunks

    def __continuous_sampling(self, tensor, target_length):
        chucks = []
        for i in range(tensor.numel() - target_length):
            if len(tensor[i:i + target_length]) == target_length:
                chucks.append(tensor[i:i + target_length])
        return chucks

    def split_list(self, lst, skip_items):
        n = len(lst)
        split1 = int(n * 0.8)
        split2 = split1 + int(n * 0.1)

        list1 = lst[:split1]  # 80%
        list2 = lst[split1 + skip_items:split2]  # 10%
        list3 = lst[split2 + skip_items:]  # 10%

        return list1, list2, list3

    def __load_from_csv(self, csv_file):
        df = pd.read_csv(csv_file, index_col=False)  # Assuming columns: person, location, data

        # Convert 'data' to a list per (person, location)
        df["out_13"] = df["out_13"].apply(self.__safe_eval)
        df["out_17"] = df["out_17"].apply(self.__safe_eval)

        # Concatenate 'out_13' and 'out_17' into a single list per person
        df["combined_out"] = df.apply(lambda row: row["out_13"][:-2] + row["out_17"][:-2], axis=1)

        if self.subtract_background_noice:
            background_noise = np.array(self.get_background_noice())
            df["combined_out"] = df["combined_out"].apply(lambda x: list(np.array(x) - background_noise))


        df["group"] = (df["name"] != df["name"].shift()) | (df["position"] != df["position"].shift())
        df["group"] = df["group"].cumsum()
        grouped = df.groupby(["name", "position", "group"])["combined_out"].apply(list).reset_index()

        pd.set_option('display.max_columns', None)

        if self.continue_sampling:
            grouped = grouped.sample(frac=1).reset_index(drop=True)

        # Convert lists to PyTorch tensors
        grouped["tensor"] = grouped["combined_out"].apply(lambda x: torch.tensor(x, dtype=torch.float32))


        # Convert all tensors into a single stacked tensor
        data_items = []

        train_items = []
        validation_items = []
        test_items = []

        labels = []

        train_labels = []
        validation_labels = []
        test_labels = []
        for _, row in grouped.iterrows():
            if int(row["position"]) in self.remove_locations or row["name"] in self.remove_names:
                continue

            if self.continue_sampling:
                splitted_items = self.__continuous_sampling(row["tensor"], self.measurements_per_sample)
            else:
                splitted_items = self.__split_up_evenly(row["tensor"], self.measurements_per_sample)

            validation_done = False
            if self.continue_sampling:
                n = len(splitted_items)
                split1 = int(n * 0.8)
                split2 = split1 + int(n * 0.1)

                for i, item in enumerate(splitted_items[:split1 - self.measurements_per_sample]):
                    if item.shape[0] == self.measurements_per_sample:
                        train_labels.append((row["name"], int(row["position"])))
                        train_items.append(item)

                for i, item in enumerate(splitted_items[split1:split2]):
                    if item.shape[0] == self.measurements_per_sample:
                        validation_labels.append((row["name"], int(row["position"])))
                        validation_items.append(item)

                for i, item in enumerate(splitted_items[split2 + self.measurements_per_sample:]):
                    if item.shape[0] == self.measurements_per_sample:
                        test_labels.append((row["name"], int(row["position"])))
                        train_items.append(item)
            else:
                for i, item in enumerate(splitted_items):
                    if item.shape[0] == self.measurements_per_sample:
                        labels.append((row["name"], int(row["position"])))
                        data_items.append(item)
                # print((row["name"], int(row["position"])), self.__trim_tensor(row["tensor"], 200))

        if self.continue_sampling:
            data_items = train_items + validation_items + test_items
            labels = train_labels + validation_labels + test_labels
            print(labels)

        data_items = torch.stack(data_items, dim=0)

        return labels, data_items

    def get_background_noice(self):
        empty_13_df = pd.read_csv(self.file_path.replace(".csv", "_background_noice_output_60Ghz_out13.csv"),
                                  index_col=False)  # Assuming columns: person, location, data

        # Convert 'data' to a list per (person, location)
        empty_13_df["out_13"] = empty_13_df["out_13"].apply(self.__safe_eval)

        # Concatenate 'out_13' and 'out_17' into a single list per person
        empty_13_df["out_13"] = empty_13_df.apply(lambda row: row["out_13"][:-2], axis=1)

        all_lists_13 = np.stack(empty_13_df["out_13"].to_list())

        # Compute element-wise average
        all_lists_13 = all_lists_13.mean(axis=0).tolist()

        empty_17_df = pd.read_csv(self.file_path.replace(".csv", "_background_noice_output_60Ghz_out17.csv"),
                                  index_col=False)  # Assuming columns: person, location, data

        # Convert 'data' to a list per (person, location)
        empty_17_df["out_17"] = empty_17_df["out_17"].apply(self.__safe_eval)

        # Concatenate 'out_13' and 'out_17' into a single list per person
        empty_17_df["out_17"] = empty_17_df.apply(lambda row: row["out_17"][:-2], axis=1)

        # Stack all lists into a 2D numpy array
        all_lists_17 = np.stack(empty_17_df["out_17"].to_list())

        # Compute element-wise average
        average_list_17 = all_lists_17.mean(axis=0).tolist()

        combined_empty = all_lists_13 + average_list_17

        return combined_empty

    def number_of_classes(self):
        return self.num_classes