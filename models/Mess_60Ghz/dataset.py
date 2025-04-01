import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast 
from collections import defaultdict

class WifiCSIDataset(Dataset):
    def __init__(self, file_path, measurements_per_sample, recreate=False):
        self.measurements_per_sample = measurements_per_sample
        if os.path.exists(file_path.replace('.csv', f'{self.measurements_per_sample}_labels.pt')) and os.path.exists(
                file_path.replace('.csv', f'{self.measurements_per_sample}_data.pt')) and not recreate:
            self.labels = torch.load(file_path.replace('.csv', f'{self.measurements_per_sample}_labels.pt'))
            self.data = torch.load(file_path.replace('.csv', f'{self.measurements_per_sample}_data.pt'))
        else:
            self.labels, self.data = self.__load_from_csv(file_path)
            torch.save(self.labels, file_path.replace('.csv', f'{self.measurements_per_sample}_labels.pt'))
            torch.save(self.data, file_path.replace('.csv', f'{self.measurements_per_sample}_data.pt'))
        self.n_samples = self.data.shape[0]

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
    
    def __load_from_csv(self, csv_file):
        df = pd.read_csv(csv_file, index_col=False)  # Assuming columns: person, location, data

        # Convert 'data' to a list per (person, location)
        print(df)
        df["out_13"] = df["out_13"].apply(self.__safe_eval)
        df["out_17"] = df["out_17"].apply(self.__safe_eval)

        # Concatenate 'out_13' and 'out_17' into a single list per person
        df["combined_out"] = df.apply(lambda row: row["out_13"][:-2] + row["out_17"][:-2], axis=1)


        df["group"] = (df["name"] != df["name"].shift()) | (df["position"] != df["position"].shift())
        df["group"] = df["group"].cumsum()
        grouped = df.groupby(["name", "position", "group"])["combined_out"].apply(list).reset_index()

        # Convert lists to PyTorch tensors
        grouped["tensor"] = grouped["combined_out"].apply(lambda x: torch.tensor(x, dtype=torch.float32))
        
        print(grouped)


        # Convert all tensors into a single stacked tensor
        data_items = []
        labels = []
        for _, row in grouped.iterrows():
            splitted_items = self.__split_up_evenly(row["tensor"], 2)
            for item in splitted_items:
                if item.shape[0] == 2:
                    labels.append((row["name"], int(row["position"])))
                    data_items.append(item)
                # print((row["name"], int(row["position"])), self.__trim_tensor(row["tensor"], 200))

        data_items = torch.stack(data_items, dim=0)
        print(data_items.shape)
        print(labels)

        return labels, data_items

    def number_of_classes(self):
        return 20