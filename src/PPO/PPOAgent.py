import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from ActorCritic import ActorCritic
from Memory import PPOMemory

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class PPOAgent:
    """Agent PPO (Proximal Policy Optimization)"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon_clip=0.2,
        k_epochs=4,
        value_coef=0.5,
        entropy_coef=0.01
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Réseau Actor-Critic
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Buffer
        self.memory = PPOMemory()
    
    def select_action(self, state):
        """Sélectionner une action"""
        action, log_prob = self.policy.get_action(state)
        
        # Obtenir aussi la valeur pour GAE
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = self.policy(state_tensor)
        
        return action, log_prob, value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        GAE réduit la variance tout en gardant un biais faible
        """
        advantages = []
        gae = 0
        
        # Calculer GAE de manière rétrograde
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            
            # A_t = δ_t + γλ A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self):
        """Mise à jour PPO"""
        # Récupérer les données du buffer
        states, actions, old_log_probs, rewards, dones, values = self.memory.get()
        
        # Calculer le next_value pour GAE (0 si terminé)
        if dones[-1]:
            next_value = 0
        else:
            state_tensor = torch.FloatTensor(states[-1]).unsqueeze(0).to(device)
            with torch.no_grad():
                _, next_value = self.policy(state_tensor)
            next_value = next_value.item()
        
        # Calculer advantages et returns avec GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normaliser les advantages (crucial pour la stabilité)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convertir en tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        
        # Optimiser la politique pendant K epochs
        for _ in range(self.k_epochs):
            # Évaluer les actions actuelles
            log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            # Ratio de probabilités (π_new / π_old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            
            # Loss totale
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, returns)
            entropy_loss = -dist_entropy.mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # Vider le buffer
        self.memory.clear()
        
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

