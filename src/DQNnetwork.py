import torch
import torch.nn as nn

class DQNnetwork(nn.module):
    def __init__(self,input_dim: int, action_dim: int,hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                     nn.GELU(),
                                     nn.Linear(hidden_dim,action_dim),
                                     nn.Softmax())
        
    def forward(self,x):
        output = self.network(x)
        return output
            