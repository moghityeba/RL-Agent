from Trainer import Trainer
import gymnasium as gym
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml

num_episodes=2000
max_timesteps=1000
update_timestep= 1024
save_freq=100

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration parameters.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


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

def train_agent(make_env, params):
    """
    Trains a PPO agent on the GridWorld environment.

    Args:
        env: The GridWorld environment instance.
        total_timesteps: The total number of training timesteps.
        grid_size: Size of gridworld for model saving.

    Returns:
        The trained PPO model.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    exp_name = params["exp_name"]
    writer = SummaryWriter(f"logs/{exp_name}")

    trainer = Trainer(make_env,writer, params)

    trainer.training_loop()

    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    trainer.plot_results()
    model_path = os.path.join(models_dir, params["model_name"])
    trainer.model.save(model_path)
    print(f"Model saved to {model_path}")

    print(trainer.test_policy(make_env))

    return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config/simple_conf.yaml",help="The path to the YAML config file for hyperparameters")
    args = parser.parse_args()

    params = load_config(args.config_path)
    train_agent(make_env, params)

if __name__ == "__main__":
    main()