import torch
import torch.nn as nn
import torch.nn.functional as F

input_dim = 45
hidden_layers1 = 1024
hidden_layers2 = 512
hidden_layers3 = 256
hidden_layers4 = 16

class DiverNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc= nn.Linear(input_dim, hidden_layers1)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_layers1)
        self.linear_fc1 = nn.Linear(hidden_layers1, hidden_layers2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_layers2)
        self.linear_fc2 = nn.Linear(hidden_layers2, hidden_layers3)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_layers3)
        self.linear_fc3 = nn.Linear(hidden_layers3, hidden_layers4)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.input_fc(x)))
        x = F.leaky_relu(self.bn2(self.linear_fc1(x)))
        x = F.leaky_relu(self.bn3(self.linear_fc2(x)))
        x = self.linear_fc3(x)
        return x


embedding_dim = 16
hidden_cls_layers1 = 32
output_dim = 4

class DiverClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(embedding_dim, hidden_cls_layers1)
        self.fc2= nn.Linear(hidden_cls_layers1, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output