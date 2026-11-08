import torch.nn as nn

class SeverityClassifier(nn.Module):
    def __init__(self, input_size, num_classes=4):
        super(SeverityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(64, 16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out = self.fc4(out)
        return out
