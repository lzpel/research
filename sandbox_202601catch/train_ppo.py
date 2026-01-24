import os
import gymnasium as gym
from catch_env import CatchEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import numpy as np

def train():
    # Directories for logs and models
    log_dir = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("out", exist_ok=True)

    # Create the environment
    def make_env():
        return CatchEnv(render_mode="rgb_array")

    env = DummyVecEnv([make_env])

    # Model definition
    # Using MlpPolicy for continuous control
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        device="auto" # Will use CUDA if available
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_catch"
    )

    # Train the agent
    total_timesteps = 100000
    print(f"Starting training for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save the final model
    model.save(os.path.join(model_dir, "ppo_catch_final"))
    print("Training complete. Model saved.")

def evaluate():
    model_path = "models/ppo_catch_final"
    if not os.path.exists(model_path + ".zip"):
        print("Model not found. Please train first.")
        return

    # Use a dummy vec env for video recorder
    def make_env():
        return CatchEnv(render_mode="rgb_array")
    
    env = DummyVecEnv([make_env])
    
    # Wrap with video recorder
    video_folder = "out/videos"
    video_length = 200
    env = VecVideoRecorder(
        env, 
        video_folder,
        record_video_trigger=lambda x: x == 0, 
        video_length=video_length,
        name_prefix="ppo_catch_evaluation"
    )

    model = PPO.load(model_path)

    print("Evaluating and recording video...")
    obs = env.reset()
    
    for _ in range(video_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
            
    # Close environment to finalize video
    env.close()
    print(f"Evaluation finished. Video saved to '{video_folder}/'.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate the trained model")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train()
