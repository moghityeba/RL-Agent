import torch.nn as nn


class DQNnetwork(nn.Module):
    def __init__(self,input_dim: int, action_dim: int,hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                     nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim, action_dim))
        
    def forward(self,x):
        output = self.network(x)
        return output
            