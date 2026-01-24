import gymnasium as gym
import numpy as np
import mujoco
import os
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont
import json

class CatchEnv(gym.Env):
    """
    MuJoCoを使用したボールキャッチ環境。
    1ステップ = 0.01秒 (MuJoCoのデフォルト timestep 0.002s * frameskip 5)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # モデルのロード
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "SO101", "scene_with_ball.xml"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # アクション空間: XMLで定義された制御範囲を持つ6つのアクチュエータ
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 観測空間: 
        # - 関節角度 (6)
        # - 関節速度 (6)
        # - ボールの位置 (3)
        # - ボールの速度 (3) 
        # - グリッパーのサイト位置 (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        self.steps = 0 # ステップカウンター
        self.max_steps = 300 # 3秒相当 (0.01s * 300)
        self.info = {}

    def _get_obs(self):
        # アームの関節角度 (6) と速度 (6)
        arm_qpos = self.data.qpos[:6]
        arm_qvel = self.data.qvel[:6]
        
        # ボールの関節位置 (7) -> 最初はpx, py, pzを取り出す
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        ball_qpos_adr = self.model.jnt_qposadr[ball_joint_id]
        ball_pos = self.data.qpos[ball_qpos_adr : ball_qpos_adr + 3].copy()
        
        ball_qvel_adr = self.model.jnt_dofadr[ball_joint_id]
        ball_vel = self.data.qvel[ball_qvel_adr : ball_qvel_adr + 3].copy()
        
        # グリッパーのサイト位置
        gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        gripper_pos = self.data.site_xpos[gripper_site_id].copy()
        
        return np.concatenate([arm_qpos, arm_qvel, ball_pos, ball_vel, gripper_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # 描画用に情報を持つ
        self.info = {
            "distance": 0.0,
            "reward": 0.0
        }
        self.steps = 0 # カウンターリセット
        
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        ball_qpos_adr = self.model.jnt_qposadr[ball_joint_id]
        ball_qvel_adr = self.model.jnt_dofadr[ball_joint_id]
        
        # ボールの初期位置をわずかにランダム化
        self.data.qpos[ball_qpos_adr] = 0.4 + self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[ball_qpos_adr + 1] = self.np_random.uniform(-0.1, 0.1)
        self.data.qpos[ball_qpos_adr + 2] = 0.05
        
        # 原点に向かう初期速度を設定
        self.data.qvel[ball_qvel_adr] = -0.5
        self.data.qvel[ball_qvel_adr + 1] = -self.data.qpos[ball_qpos_adr + 1] * 2.0
        
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.data.ctrl[:] = action
        
        # フレームスキップ (1ステップ = 0.002s * 5 = 0.01s)
        frameskip = 5
        for _ in range(frameskip):
            mujoco.mj_step(self.model, self.data)
            
        observation = self._get_obs()
        
        # 報酬計算
        ball_pos = observation[12:15]
        gripper_pos = observation[18:21]
        dist = np.linalg.norm(ball_pos - gripper_pos)
        
        # 1. 距離報酬: 近づくほど報酬が高くなる (0.0 から 1.0 の範囲で変化)
        reward = np.exp(-5.0 * dist) 
        
        # 2. キャッチ成功ボーナス
        if dist < 0.04:
            reward += 10.0
            
        # 3. 動作の滑らかさペナルティ (アクションが大きすぎるとマイナス)
        # reward -= 0.1 * np.mean(np.square(action))
            
        self.steps += 1
            
        terminated = False
        # ボールが床付近（z < 0.03）まで落ちたか、外に飛んでいった場合
        if ball_pos[2] < 0.03 or np.linalg.norm(ball_pos[:2]) > 2.0: 
            terminated = True
            reward -= 5.0
            
        # 3秒経過したら打ち切り
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        self.info = {
            "distance": float(dist),
            "reward": float(reward)
        }
        
        return observation, reward, terminated, truncated, self.info

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            ret = self.renderer.render()
            # 報酬とグリッパー・ボール間の距離を表示
            # 辞書をインデント付きの文字列に変換
            info_str = json.dumps(self.info, indent=4)
            ret = add_text_to_frame(ret, info_str)
            return ret

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
def add_text_to_frame(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((10,10), text, fill=(255,255,255))
    frame = np.array(img)
    return frame