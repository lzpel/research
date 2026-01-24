import os
import gymnasium as gym
from catch_env import CatchEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecVideoRecorder
import numpy as np

def train():
    # ログとモデルの保存ディレクトリ
    log_dir = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("out", exist_ok=True)

    # 環境の作成用関数
    def make_env():
        return CatchEnv(render_mode="rgb_array")

    # 並列処理用の環境（SubprocVecEnv）
    env = SubprocVecEnv([make_env])

    # モデルの定義
    # 連続値制御のため MlpPolicy を使用
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
        device="auto" # 利用可能な場合はCUDA（GPU）を自動使用
    )

    # コールバック（学習途中のモデル保存など）
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_catch"
    )

    # 学習の実行
    total_timesteps = 100000
    print(f"学習を開始します（計 {total_timesteps} ステップ）...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # 最終モデルの保存
    model.save(os.path.join(model_dir, "ppo_catch_final"))
    print("学習が完了しました。モデルを保存しました。")

def evaluate():
    model_path = "models/ppo_catch_final"
    if not os.path.exists(model_path + ".zip"):
        print("モデルが見つかりません。先に学習を実行してください。")
        return

    # 評価用には DummyVecEnv を使用
    def make_env():
        return CatchEnv(render_mode="rgb_array")
    
    env = DummyVecEnv([make_env])
    
    # ビデオ録画用のラッパー
    video_folder = "out/videos"
    video_length = 300 # 3秒分（1ステップ0.01秒 × 300）
    env = VecVideoRecorder(
        env, 
        video_folder,
        record_video_trigger=lambda x: x == 0, 
        video_length=video_length,
        name_prefix="ppo_catch_evaluation"
    )

    model = PPO.load(model_path)

    print("評価を開始し、動作を録画します...")
    obs = env.reset()
    
    for _ in range(video_length):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
            
    # 環境を閉じてビデオを確定
    env.close()
    print(f"評価が完了しました。ビデオは '{video_folder}/' に保存されました。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="学習済みモデルの評価を実行")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    else:
        train()
