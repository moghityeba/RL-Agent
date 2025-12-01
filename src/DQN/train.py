import torch
import torch.nn as nn
import gymnasium as gym
from DQNagent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt


num_episodes = 500
max_steps = 1000
env = gym.make("LunarLander-v3")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )

# Métriques
episode_rewards = []
episode_lengths = []
losses = []

for episode in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    episode_loss = []
    
    for step in range(max_steps):
        # Sélectionner et exécuter une action
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Stocker dans le replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Mettre à jour le réseau
        loss = agent.update()
        if loss is not None:
            episode_loss.append(loss)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    # Décroissance d'epsilon
    agent.decay_epsilon()
    agent.step_scheduler()
    # Sauvegarder les métriques
    episode_rewards.append(episode_reward)
    episode_lengths.append(step + 1)
    if episode_loss:
        losses.append(np.mean(episode_loss))
        
    # Afficher les progrès
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}/{num_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)}")
        
    # Critère de succès (LunarLander est résolu avec reward >= 200)
    if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 200:
        print(f"\nEnvironnement résolu en {episode + 1} épisodes!")
        break
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
# Rewards
ax1.plot(episode_rewards, alpha=0.6, label='Episode Reward')

# Moving average
if len(episode_rewards) >= 10:
    moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    ax1.plot(range(9, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')

ax1.axhline(y=200, color='g', linestyle='--', label='Target (200)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Rewards')
ax1.legend()
ax1.grid(True)

# Losses
if losses:
    ax2.plot(losses, alpha=0.6)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True)

plt.tight_layout()
plt.savefig('dqn_training_results.png')
plt.show()
env.close()

torch.save(agent.q_network.state_dict(), '/Users/moghityebari/Desktop/Personal_projects/RL_Agent/RL-Agent/models/dqn_lunarlander.pth')