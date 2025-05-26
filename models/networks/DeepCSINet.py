import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)

class DeepCSINet(nn.Module):
    def __init__(self, input_channels=60, num_classes=20, dropout_rate=0.3, kernel_size=3, device='cuda'):
        super(DeepCSINet, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.rb1 = nn.Sequential(
            ConvBlock(128, 128, kernel_size, dropout_rate),
            ConvBlock(128, 128, kernel_size, dropout_rate)
        )
        self.rb2 = nn.Sequential(
            ConvBlock(128, 128, kernel_size, dropout_rate),
            ConvBlock(128, 128, kernel_size, dropout_rate)
        )
        self.rb3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size, dropout_rate),
            ConvBlock(256, 256, kernel_size, dropout_rate)
        )
        self.rb4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size, dropout_rate),
            ConvBlock(512, 512, kernel_size, dropout_rate)
        )

        # Output head
        self.location_head = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        self.to(device)

    def forward(self, x):
        # Input shape: [batch_size, time_steps, features]
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim

        x = x.permute(0, 2, 1)  # to [batch_size, channels, time_steps]

        x = self.init_conv(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        location = self.location_head(x)
        return location
