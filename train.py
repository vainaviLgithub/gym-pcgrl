import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from callbacks import CustomLoggerCallback

from utils import make_env

# --- Configuration ---
log_dir = "./taxi_logs"
os.makedirs(log_dir, exist_ok=True)
total_timesteps = 1_000_000  # Longer training for better convergence
seed = 42

# --- Create Vectorized & Normalized Environments ---
def create_env(seed):
    env = DummyVecEnv([make_env(log_dir=log_dir, seed=seed)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    return env

env = create_env(seed)
eval_env = create_env(seed + 100)  # Different seed for eval

# --- Callbacks ---
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=log_dir,
    name_prefix="ppo_taxi"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=25000,
    deterministic=True,
    render=False
)

# --- Custom Policy Architecture ---
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
)

# --- PPO Model ---
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    clip_range=0.2,
    max_grad_norm=0.5,
    target_kl=0.03,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=log_dir,
    seed=seed,
)

# --- Train the model ---
custom_logger = CustomLoggerCallback(log_dir=log_dir, verbose=1)

model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, eval_callback, custom_logger]
)

# --- Save the final model ---
model.save(os.path.join(log_dir, "final_model"))

# --- Final Evaluation ---
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"\n✅ Final Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
