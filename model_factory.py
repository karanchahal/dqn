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

def get_model(params: Params):

    if params.network_type == "mlp":
        return MLP(params)
