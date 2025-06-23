from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gym_pcgrl.envs.pcgrl_env import GridGameEnv
import os


def make_env(log_dir=None, render=False, seed=None, normalize=False):
    """
    Creates and returns a PCGRL GridGameEnv instance wrapped with a Monitor for logging and optional normalization.

    Parameters:
        log_dir (str): Directory where logs will be saved. If None, logging is disabled.
        render (bool): Whether to enable GUI rendering (not recommended during training).
        seed (int): Seed for reproducibility.
        normalize (bool): Whether to normalize the environment's observations and rewards.

    Returns:
        function: An environment initializer (for DummyVecEnv or SubprocVecEnv)
    """
    def _init():
        env = GridGameEnv(config_file="game_levels.json")

        if render:
            env.render_mode = "human"

        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

        if log_dir:
            env = Monitor(env, log_dir)

        return env  # ← DON'T wrap in DummyVecEnv here

    return _init


class RenderMonitor(Monitor):
    """
    A monitor wrapper that optionally renders the environment after each step.
    
    Parameters:
        env: The environment to wrap.
        log_dir: Directory to save monitor logs.
        render (bool): If True, renders the environment each step.
    """
    def __init__(self, env, log_dir=None, render=False):
        self.render_gui = render
        log_dir = os.path.join(log_dir, "monitor") if log_dir else None
        super().__init__(env, log_dir)

    def step(self, action):
        if self.render_gui:
            self.render()
        return super().step(action)

def load_model(log_dir):
    """
    Loads the best model from the specified directory.
    
    Parameters:
        log_dir (str): Directory containing the saved model.

    Returns:
        PPO: The loaded PPO model.
    """
    model_path = os.path.join(log_dir, "best_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return PPO.load(model_path)
