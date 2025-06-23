import json
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # No VecNormalize for now
from gym_pcgrl.envs.pcgrl_env import GridGameEnv
from datetime import datetime

# Paths
model_path = "./taxi_logs/final_model.zip"
norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

# Load model
model = PPO.load(model_path)

# Load game levels
with open("game_levels.json", "r") as f:
    all_levels = json.load(f)

# Evaluation function
def evaluate_on_levels(model, levels):
    results = {}
    detailed_log = {}

    for difficulty in ["easy", "medium", "hard"]:
        filtered_levels = [lvl for lvl in levels if lvl["difficulty"] == difficulty]
        if not filtered_levels:
            print(f"⚠️ No levels found for difficulty: {difficulty}")
            continue

        print(f"\n🔍 Evaluating difficulty: {difficulty}")
        env = DummyVecEnv([lambda: GridGameEnv(levels=filtered_levels, use_curriculum=False)])

        total_rewards = []
        per_level_log = []

        for i, _ in enumerate(filtered_levels):
            obs = env.reset()
            done = [False]
            total_reward = 0

            prev_pos = None
            stuck_counter = 0

            while not done[0]:
                # Get agent position
                try:
                    pos = env.get_attr("get_agent_position")[0]()
                except Exception:
                    pos = None

                if pos == prev_pos:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                prev_pos = pos

                # If stuck, pick a random action
                if stuck_counter >= 50:
                    action = [env.action_space.sample()]
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]

            print(f"   🧪 Level {i + 1}/{len(filtered_levels)} — Reward: {total_reward:.2f}")
            total_rewards.append(total_reward)
            per_level_log.append({"level_index": i, "reward": total_reward})

        avg_reward = np.mean(total_rewards)
        print(f"✅ Avg reward for {difficulty}: {avg_reward:.2f}")
        results[difficulty] = avg_reward
        detailed_log[difficulty] = per_level_log
        env.close()

    return results, detailed_log

# Run evaluation
print("🔍 Evaluating PPO agent per difficulty:")
avg_result, log_result = evaluate_on_levels(model, all_levels)

# Save results
eval_data = {
    "average_results": avg_result,
    "detailed_log": log_result
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"eval_results_{timestamp}.json"
with open(save_path, "w") as f:
    json.dump(eval_data, f, indent=4)

print(f"\n📁 Saved detailed evaluation results to: {save_path}")
print("\n📊 Final Results:")
for difficulty, avg_reward in avg_result.items():
    print(f" - {difficulty}: {avg_reward:.2f}")
