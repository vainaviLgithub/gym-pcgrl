import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    rewards = ea.Scalars("rollout/ep_rew_mean")
    timesteps = [r.step for r in rewards]
    values = [r.value for r in rewards]

    return timesteps, values

def plot_rewards(log_dir):
    timesteps, rewards = load_tensorboard_data(log_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards, label='Mean Episode Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('PPO Training Progress')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_reward_plot.png")
    plt.show()

if __name__ == "__main__":
    log_path = "./taxi_logs/PPO_12"  # change this to your latest run
    plot_rewards(log_path)
