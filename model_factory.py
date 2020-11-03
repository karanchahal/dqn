import torch.nn as nn
from structures import Params
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, params: Params, hidden_dim=50):
        super().__init__()
        self.fc1 = nn.Linear(in_features=params.state_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=params.action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class CNNNet(nn.Module):

    def __init__(self, params: Params):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(num_features=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(25, 128)
        self.fc2 = nn.Linear(128, params.action_dim)
    
    def forward(self, x):

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



def get_model(params: Params):

    if params.network_type == "mlp":
        return MLP(params)

    if params.network_type == "cnn":
        return CNNNet(params)
