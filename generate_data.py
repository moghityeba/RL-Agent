import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3")
observation, info = env.reset()

episodes_data = []
for episode in range(100):
    obs, _ = env.reset()
    episode_buffer = []
    
    done = False
    i = 0
    while not done:
        action = env.action_space.sample()  # Replace with your policy
        print(f'Action {i}: ',action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        episode_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': terminated or truncated
        })
        print(f'Observation{i}: ',episode_buffer[-1])
        obs = next_obs
        done = terminated or truncated
        print("-"*50)
        i+=1
    episodes_data.append(episode_buffer)

print(env.spec.max_episode_steps) 
# Save your dataset
#np.save('atari_breakout_data.npy', episodes_data, allow_pickle=True)