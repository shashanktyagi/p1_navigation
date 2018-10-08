import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(self.state_size, 64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()

        self.out = nn.Linear(128, self.action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        x = self.fc1(state)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.out(x)

        return x
