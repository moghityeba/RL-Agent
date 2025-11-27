import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 64):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialisation orthogonale (meilleure pour RL)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
            
    def forward(self, state):
        """Retourne (action_probs, state_value)"""
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def get_action(self, state):
        """Sélectionner une action selon la politique"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = self.forward(state)
        
        # Échantillonner une action
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action.item(), action_log_prob.item()
    
    def evaluate(self, states, actions):
        """Évaluer les actions (pour l'update)"""
        action_probs, state_values = self.forward(states)
        
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_values.squeeze(), dist_entropy