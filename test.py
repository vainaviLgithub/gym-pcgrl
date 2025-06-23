# import time
# import numpy as np
# from gym_pcgrl.envs.pcgrl_env import GridGameEnv

# # Initialize environment
# env = GridGameEnv(config_file="game_levels.json")

# # Reset and get initial observation
# observation = env.reset()
# print("Initial Observation:")
# env.render()

# # Run a test episode with random actions
# done = False
# step_count = 0

# while not done and step_count < 20:  # Limit steps to prevent infinite loops
#     action = np.random.choice([0, 1, 2, 3])  # Randomly choose an action (Up, Down, Left, Right)
#     obs, reward, done, _ = env.step(action)

#     print(f"Step {step_count+1}: Action {action}, Reward: {reward}")
#     env.render()
    
#     step_count += 1
#     time.sleep(0.5)  # Pause for readability

# print("Test episode finished.")

# import json

# def print_grid(level):
#     for row in level["grid"]:
#         print(" ".join(row))
#     print(f"\nTaxi: {level['taxi']}, Passenger: {level['passenger']}, Dropoff: {level['dropoff']}\n")

# # Test by loading and printing one level
# with open("game_levels.json", "r") as f:
#     levels = json.load(f)

# print_grid(levels[0])  # Print the first generated level

import os
import gymnasium as gym
from stable_baselines3 import PPO
from utils import make_vec_env

log_dir = "./taxi_logs"
model_path = os.path.join(log_dir, "latest_model.zip")

def test_taxi(num_episodes=5, render=True):
    """
    Tests the trained PPO model on Taxi-v3.
    """
    if not os.path.exists(model_path):
        print("No trained model found! Train the model first.")
        return
    
    env = gym.make("Taxi-v3",render_mode = "ansi")
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"\n🚖 Episode {episode + 1} 🚖")
        
        while not done:
            if render:
                env.render()
                
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())  # Unpack correctly
            done = terminated or truncated  # Ensure episode ends correctly
            total_reward += reward
        
        print(f"✅ Episode {episode + 1} finished with total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    test_taxi(num_episodes=5, render=True)
