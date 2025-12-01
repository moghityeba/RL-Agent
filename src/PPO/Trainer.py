import torch
import numpy as np
from ActorCritic import ActorCritic
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def make_env(env_id="LunarLander-v3", seed=None, idx=0):
    """
    Crée une FACTORY qui retourne un environnement
    
    Returns:
        callable: Une fonction qui crée un environnement quand appelée
    """
    def _init():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed + idx)
        return env
    return _init

class Trainer:
    """
    Trainer class for running PPO training on a Gym environment.
    """
    
    def __init__(
        self,
        make_env,
        writer,
        params
    ):
        self.env_id = "LunarLander-v3"
        self.gamma = params["gamma"]
        self.gae_lambda = params["gae_lambda"]
        self.epsilon_clip = params["epsilon_clip"]
        self.n_epochs = params["n_epochs"]
        self.value_coef = params["value_coef"]
        self.entropy_coef = params["entropy_coef"]
        self.seed = params["seed"]
        self.n_envs = params["n_envs"]
        self.num_steps = params["num_steps"]
        self.batch_size = self.n_envs*self.num_steps
        self.mini_batch_size = self.batch_size//params["num_minibatches"]
        self.update = (params["total_timestamp"]+self.batch_size-1)//self.batch_size
        self.warmup_updates = params.get("warmup_updates", 10) 
        # Réseau Actor-Critic
        self.env = self._make_vec_env()
        self.state_dim = self.env.single_observation_space.shape[0]
        self.action_dim = self.env.single_action_space.n
        self.model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.update - self.warmup_updates,      
            eta_min=2e-6   
        )
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.obs = self.to_tensor(self.env.reset()[0]).to(self.device)
        self.episode_returns = []
        self.episode_lengths = []
        self.current_episode_returns = np.zeros(self.n_envs)
        self.current_episode_lengths = np.zeros(self.n_envs)
        self.writer = writer
        self.loss_step = 0
        self.reward_step = 0
        self.global_step = 0
        self.anneal_lr = params["anneal_lr"]
        self.lr = params["learning_rate"]
        self.model_name = params["model_name"]
        self.min_lr = params.get("min_lr", 1e-7)  # LR minimum
        self.max_lr = params["learning_rate"] 
        
    def _make_vec_env(self):
        """Crée les environnements vectorisés"""
        def make_env(rank):
            def _init():
                env = gym.make(self.env_id)
                env.reset(seed=self.seed + rank if self.seed is not None else None)
                return env
            return _init
        
        return AsyncVectorEnv([
            make_env(i) for i in range(self.n_envs)
        ])
        
    def to_tensor(self,arr: np.ndarray) -> torch.Tensor:
        """
        Convert an array-like object to a PyTorch tensor.

        Args:
            arr: Input array.

        Returns:
            torch.Tensor: Converted tensor.
        """
        return torch.tensor(arr, dtype=torch.float)
    
    @torch.no_grad()
    def sample(self) -> dict:
        """
        Collect samples from the environment using the current policy.

        Returns:
            dict: Dictionary containing observations, actions, values, log probabilities,
                  advantages, and rewards.
        """
        rewards = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        actions = torch.zeros((self.n_envs, self.num_steps), dtype=torch.long)
        dones = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)
        observations = torch.zeros((self.n_envs, self.num_steps, self.state_dim), dtype=torch.float)
        values = torch.zeros((self.n_envs, self.num_steps+1), dtype=torch.float)
        log_probs = torch.zeros((self.n_envs, self.num_steps), dtype=torch.float)

        for t in range(self.num_steps):
            observations[:,t] = self.obs
            action_distribution,v = self.model(self.obs)
            action = action_distribution.sample()
            actions[:,t] = action
            values[:,t] = v.reshape(self.n_envs,).detach()
            log_probs[:,t] = action_distribution.log_prob(action).detach()
            self.obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
            done = terminated | truncated
            self.obs = self.to_tensor(self.obs).to(self.device)
            dones[:,t] = self.to_tensor(done)
            rewards[:,t] = self.to_tensor(reward)
            
            # Log episode rewards and lengths if available
            for i in range(self.n_envs):
                self.current_episode_returns[i] += reward[i]
                self.current_episode_lengths[i] += 1
                
                if done[i]:
                    # Épisode terminé
                    episode_return = self.current_episode_returns[i]
                    episode_length = self.current_episode_lengths[i]
                    
                    self.episode_returns.append(episode_return)
                    self.episode_lengths.append(episode_length)
                    
                    # Log dans TensorBoard
                    self.writer.add_scalar("episode/return", episode_return, self.global_step)
                    self.writer.add_scalar("episode/length", episode_length, self.global_step)
                    
                    # Reset counters
                    self.current_episode_returns[i] = 0
                    self.current_episode_lengths[i] = 0
                        
            self.global_step+=1
            
        # Get value for the final observation
        _, v = self.model(self.obs)
        values[:,self.num_steps] = v.reshape(self.n_envs)
        advantages = self.GAE(values, rewards, dones)

        return {
            'observations': observations.reshape(self.batch_size, *observations.shape[2:]),
            'actions': actions.reshape(self.batch_size, *actions.shape[2:]),
            'values': values[:,:-1].reshape(self.batch_size, *values.shape[2:]),
            'log_prob': log_probs.reshape(self.batch_size, *log_probs.shape[2:]),
            'advantages': advantages.reshape(self.batch_size, *advantages.shape[2:]),
            'rewards': rewards.reshape(self.batch_size, *advantages.shape[2:])
        }
        
    def train(self, samples: dict) -> None:
        """
        Train the model for a fixed number of epochs using mini-batches.

        Args:
            samples (dict): Dictionary containing training samples.
        """
        for _ in range(self.n_epochs):
            idx = torch.randperm(self.batch_size)
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mini_batch_idx = idx[start:end]
                mini_batch_samples = {
                        k: v[mini_batch_idx].to(self.device) for k,v in samples.items()
                    }
                loss = self.compute_loss(mini_batch_samples)
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                # Calculate and log gradient norm
                total_norm = sum(
                    param.grad.data.norm(2).item() ** 2 for param in self.model.parameters() if param.grad is not None
                ) ** 0.5
                self.writer.add_scalar(
                    "grad_norm",
                    total_norm, 
                    global_step = self.loss_step-1
                )


                self.optimizer.step()
    
    def compute_loss(self, samples: dict) -> torch.Tensor:
        """
        Compute the PPO loss for a mini-batch.

        Args:
            samples (dict): Mini-batch samples.

        Returns:
            torch.Tensor: Computed loss.
        """
        sample_ret = samples["values"]+samples["advantages"]
        old_values = samples["values"]
        action_distribution,values = self.model(samples["observations"])
        values = values.squeeze(1)
        log_probs = action_distribution.log_prob(samples["actions"])
        advantages = samples["advantages"]
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        value_f = (sample_ret-values)**2
        value_pred_clipped = (
            torch.clamp(
                values-old_values, 
                -self.epsilon_clip, 
                self.epsilon_clip
            ) + old_values
        )
        value_f_clipped = (value_pred_clipped - sample_ret)**2
        
        loss = (
            -self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean()
            + self.value_coef * 0.5 * (torch.max(value_f, value_f_clipped)).mean()
            - self.entropy_coef * action_distribution.entropy().mean() 
        )
        self.writer.add_scalar("global_loss",loss, global_step = self.loss_step)
        self.writer.add_scalar(
            "policy_loss",
            self.ppo_clip(log_probs, samples["log_prob"], adv_norm).mean(), 
            global_step = self.loss_step
        )
        self.writer.add_scalar(
            "value_loss",
            ((sample_ret-values)**2).mean(), 
            global_step = self.loss_step
        )
        self.writer.add_scalar(
            "entropy_loss",
            action_distribution.entropy().mean() , 
            global_step = self.loss_step
        )
        self.loss_step+=1

        return loss
    
    def GAE(self, values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute the Generalized Advantage Estimation (GAE).

        Args:
            values (torch.Tensor): Estimated state values.
            rewards (torch.Tensor): Rewards collected.
            dones (torch.Tensor): Binary flags indicating episode termination.

        Returns:
            torch.Tensor: Computed advantages.
        """

        advantages = torch.zeros_like(rewards)
        last_advantages = torch.zeros(self.n_envs)
        for t in reversed(range(self.num_steps)):
            delta = rewards[:,t] + self.gamma * values[:,t+1] * (1.0 - dones[:,t]) - values[:,t]
            advantages[:,t] = delta + self.gamma * self.gae_lambda * (1.0 - dones[:,t]) * last_advantages
            last_advantages = advantages[:, t]
        return advantages

    
    def ppo_clip(self,log_prob: torch.Tensor, log_prob_old: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Calculate the PPO clipped objective.

        Args:
            log_prob (torch.Tensor): New log probabilities.
            log_prob_old (torch.Tensor): Old log probabilities.
            advantages (torch.Tensor): Advantage estimates.

        Returns:
            torch.Tensor: Clipped PPO loss.
        """
        ratio = torch.exp(log_prob-log_prob_old)
        loss = ratio * advantages
        loss_clip = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip) * advantages
        return torch.min(loss, loss_clip)

    def training_loop(self):
        """
        Main training loop.
        """
        for e in tqdm(range(1,self.update+1)):
            if self.anneal_lr:
                coeff = 1 - (e/self.update)
                self.optimizer.param_groups[0]["lr"] = coeff * self.lr
                
            samples = self.sample()
            self.train(samples)
            self._update_learning_rate(e)
            self.reward_step += 1
            # Logging périodique
            if e % 10 == 0:
                if len(self.episode_returns) > 0:
                    recent = self.episode_returns[-min(100, len(self.episode_returns)):]
                    avg = np.mean(recent)
                    print(f"\nUpdate {e}: Avg return (last {len(recent)}): {avg:.2f}")
                
    
    def _update_learning_rate(self, update_num):
        """
        Update learning rate avec warmup phase
        
        Args:
            update_num: Current update number (1-indexed)
        """
        if update_num <= self.warmup_updates:
            # ✅ WARMUP PHASE: Linear warmup de min_lr à max_lr
            progress = update_num / self.warmup_updates
            current_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.writer.add_scalar("train/learning_rate", current_lr, update_num)
            
            if update_num % 5 == 0 or update_num == self.warmup_updates:
                print(f"[Warmup {update_num}/{self.warmup_updates}] LR: {current_lr:.6f}")
                
                
    @torch.no_grad()
    def test_policy(self,make_env, n_eval_episodes: int = 10) -> tuple:
        """
        Evaluate the current policy over several episodes.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.

        Returns:
            tuple: Mean and standard deviation of rewards over the episodes.
        """
        env = gym.make("LunarLander-v3")
        rewards = [0]*n_eval_episodes
        for n in range(n_eval_episodes):
            observation,_ = env.reset()
            done = False
            while not done:
                obs = self.to_tensor(observation).reshape(1,-1).to(self.device)
                action_distribution, _ = self.model(obs)
                a = torch.argmax(action_distribution.logits).item()
                observation, reward, done, truncated, _ = env.step(a)
                done = done or truncated
                rewards[n] += reward
        return np.mean(rewards), np.std(rewards)
    
    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # Rewards
        ax1.plot(self.episode_returns, alpha=0.6, label='Episode Reward')

        # Moving average
        if len(self.episode_returns) >= 10:
            moving_avg = np.convolve(self.episode_returns, np.ones(10)/10, mode='valid')
            ax1.plot(range(9, len(self.episode_returns)), moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')

        ax1.axhline(y=200, color='g', linestyle='--', label='Target (200)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True)


        plt.tight_layout()
        plt.savefig('dqn_training_results.png')
        plt.show()

