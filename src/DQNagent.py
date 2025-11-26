import torch
from DQNnetwork import DQNnetwork
from Replaybuffer import ReplayBuffer
import random
import torch.nn as nn


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DQNAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate = 1e-3,
                 gamma = 0.99,
                 epsilon_start = 1.0,
                 epsilon_end = 0.01,
                 epsilon_decay = 0.95,
                 buffer_size = 10000,
                 batch_size = 64,
                 target_update_freq = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Réseaux network
        self.q_network = DQNnetwork(self.state_dim,self.action_dim)
        self.target_network = DQNnetwork(self.state_dim,self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),lr = learning_rate)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.update_count = 0
        
    def select_action(self,state,training = True):
        if training and random.random() < self.epsilon:
            return random.randint(0,self.action_dim-1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
        
    def update(self):
        
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards,next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        current_q_values = self.q_network(states)
        
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma*next_q_values*(1 - dones)
            
        loss = nn.MSELoss()(current_q_values, target_q_values)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Mise à jour du target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()