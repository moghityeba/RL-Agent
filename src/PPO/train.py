import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from ActorCritic import ActorCritic
from Memory import PPOMemory
from PPOAgent import PPOAgent
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


num_episodes=2000
max_timesteps=1000
update_timestep= 1024
save_freq=100

env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    epsilon_clip=0.3,
    k_epochs=4,
    value_coef=0.5,
    entropy_coef=0.02
)

# Métriques
episode_rewards = []
actor_losses = []
critic_losses = []
entropy_losses = []

print("🚀 Début de l'entraînement PPO sur LunarLander...\n")

timestep = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    for t in range(max_timesteps):
        timestep += 1
        
        # Sélectionner une action
        action, log_prob, value = agent.select_action(state)
        
        # Exécuter l'action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Stocker dans le buffer
        agent.memory.add(state, action, log_prob, reward, done, value)
        
        episode_reward += reward
        state = next_state
        
        # Update PPO après update_timestep steps
        if timestep % update_timestep == 0:
            actor_loss, critic_loss, entropy_loss = agent.update()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_losses.append(entropy_loss)
        
        if done:
            break
    
    episode_rewards.append(episode_reward)
    
    # Afficher les progrès
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Last Reward: {episode_reward:7.2f} | "
                f"Timesteps: {timestep:6d}")
    
    
    # Critère de succès
    if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 200:
        print(f"\n✅ Environnement résolu en {episode + 1} épisodes!")
        print(f"Moyenne sur 100 épisodes: {np.mean(episode_rewards[-100:]):.2f}")
        break

env.close()
    


    
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Rewards
ax1 = axes[0, 0]
ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')

if len(episode_rewards) >= 10:
    ma10 = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    ax1.plot(range(9, len(episode_rewards)), ma10, 'r-', linewidth=2, label='MA(10)')

if len(episode_rewards) >= 100:
    ma100 = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
    ax1.plot(range(99, len(episode_rewards)), ma100, 'g-', linewidth=2, label='MA(100)')

ax1.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Target')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Rewards (PPO)')
ax1.legend()
ax1.grid(True)

# Actor Loss
ax2 = axes[0, 1]
if actor_losses:
    ax2.plot(actor_losses, alpha=0.6, label='Actor Loss')
    if len(actor_losses) >= 50:
        ma = np.convolve(actor_losses, np.ones(50)/50, mode='valid')
        ax2.plot(range(49, len(actor_losses)), ma, 'r-', linewidth=2, label='MA(50)')
    ax2.set_xlabel('Update')
    ax2.set_ylabel('Loss')
    ax2.set_title('Actor Loss')
    ax2.legend()
    ax2.grid(True)

# Critic Loss
ax3 = axes[1, 0]
if critic_losses:
    ax3.plot(critic_losses, alpha=0.6, label='Critic Loss')
    if len(critic_losses) >= 50:
        ma = np.convolve(critic_losses, np.ones(50)/50, mode='valid')
        ax3.plot(range(49, len(critic_losses)), ma, 'r-', linewidth=2, label='MA(50)')
    ax3.set_xlabel('Update')
    ax3.set_ylabel('Loss')
    ax3.set_title('Critic Loss')
    ax3.legend()
    ax3.grid(True)

# Entropy Loss
ax4 = axes[1, 1]
if entropy_losses:
    ax4.plot(entropy_losses, alpha=0.6, label='Entropy')
    if len(entropy_losses) >= 50:
        ma = np.convolve(entropy_losses, np.ones(50)/50, mode='valid')
        ax4.plot(range(49, len(entropy_losses)), ma, 'r-', linewidth=2, label='MA(50)')
    ax4.set_xlabel('Update')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy (Exploration)')
    ax4.legend()
    ax4.grid(True)

plt.tight_layout()
plt.savefig('ppo_training_results.png', dpi=150)
plt.show()