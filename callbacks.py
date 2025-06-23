from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import csv

class CustomLoggerCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.csv_file = os.path.join(log_dir, "episode_rewards.csv")

        # Write CSV header
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward", "Timesteps"])

    def _on_step(self) -> bool:
        done_array = self.locals["dones"]
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]

        for i, done in enumerate(done_array):
            if done:
                episode_reward = infos[i].get("episode", {}).get("r")
                if episode_reward is not None:
                    self.episode_rewards.append(episode_reward)
                    step_count = self.num_timesteps
                    episode_num = len(self.episode_rewards)

                    # Log to CSV
                    with open(self.csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([episode_num, episode_reward, step_count])

                    if self.verbose > 0:
                        print(f"Episode {episode_num} - Reward: {episode_reward}, Timesteps: {step_count}")
        return True
