import torch


def coord_to_label(coord, grid_size=4):
    x = coord[0][0].item()
    y = coord[0][1].item()
    return int(y * grid_size + x)  # Compute label


def label_to_coord(label, grid_size=4):
    x = label % grid_size  # Column index
    y = label // grid_size  # Row index
    return torch.tensor([x, y], dtype=torch.float32)