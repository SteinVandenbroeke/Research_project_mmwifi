import torch.nn as nn

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size + 500)
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size + 500, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, hidden_size - 500)
        self.dropout3 = nn.Dropout(0.5)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size - 500, num_classes)
        self.cuda(device=device)

    def forward(self, x):
        out = self.l1(x)
        out = self.dropout1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.dropout3(out)
        out = self.relu3(out)
        out = self.l4(out)
        # no activation and no softmax at the end
        return out
