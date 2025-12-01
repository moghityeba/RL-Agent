import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initialize a given layer using orthogonal initialization for the weights
    and constant initialization for the biases.

    Args:
        layer (nn.Module): The neural network layer to initialize.
        std (float): The standard deviation (gain) for orthogonal initialization. 
                               Default is np.sqrt(2).
        bias_const (float): The constant value to initialize biases. Default is 0.0.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor with separate layers for more flexibility
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_ln1 = nn.LayerNorm(hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_ln2 = nn.LayerNorm(hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # Critic
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_ln1 = nn.LayerNorm(hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_ln2 = nn.LayerNorm(hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.Tanh()
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
            
    def forward(self, state):
        # Actor
        x = self.actor_fc1(state)
        x = self.actor_ln1(x)
        x = self.activation(x)
        x = self.actor_fc2(x)
        x = self.actor_ln2(x)
        x = self.activation(x)
        action_logits = self.actor_out(x)
        
        # Critic
        v = self.critic_fc1(state)
        v = self.critic_ln1(v)
        v = self.activation(v)
        v = self.critic_fc2(v)
        v = self.critic_ln2(v)
        v = self.activation(v)
        state_value = self.critic_out(v)
        
        action_distribution = Categorical(logits=action_logits)
        
        return action_distribution, state_value
    
    def save(self, path: str) -> None:
        """
        Save the model state dictionary to a file.

        Args:
            path (str): The path where the state dictionary should be saved.
        """
        torch.save(self.state_dict(), path)