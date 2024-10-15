import torch
from torch.nn import Module
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoveNet(Module):
    def __init__(self, n_state, n_action, n_hidden=256, lr=0.005):
        super(MoveNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action),
            torch.nn.Tanh()
            )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)

    def forward(self, state):
        return self.net(state)

    def update(self, state, action):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        action = torch.tensor(action, dtype=torch.float32, device=DEVICE)
        action_by_net = self.forward(state)
        loss = self.criterion(action_by_net, action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = self.forward(state)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, -1.0, 1.0)
        return action
    
        
