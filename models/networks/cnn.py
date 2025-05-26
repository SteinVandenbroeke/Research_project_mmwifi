import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepTimeSeriesCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate, kernel_size, device):
        super(DeepTimeSeriesCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=128, kernel_size=kernel_size),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after first conv

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after second conv

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output shape: (batch_size, 128, 1)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),           # Shape: (batch_size, 128)
            nn.Dropout(dropout_rate),  # Extra dropout before FC
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        self.cuda(device=device)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            # x shape: (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)  # Convert to (batch_size, features, time_steps)
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
