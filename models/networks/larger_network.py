import torch.nn as nn

# Fully connected neural network with one hidden layer
class LargerNetwork(nn.Module):
    def __init__(self, input_size, num_classes, device):
        super(LargerNetwork, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 500)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(500, 1200)
        self.dropout3 = nn.Dropout(0.5)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(1200, 500)
        self.dropout4 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(500, num_classes)
        self.size = 7
        self.cuda(device=device)

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        out = self.l1(x)
        # out = self.dropout1(out)
        # out = self.relu1(out)
        # out = self.l2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.dropout3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.dropout4(out)
        out = self.relu4(out)
        out = self.l5(out)
        # no activation and no softmax at the end
        return out
