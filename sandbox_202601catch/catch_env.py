import gymnasium as gym
import numpy as np
import mujoco
import os
from gymnasium import spaces

class CatchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Load model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "SO101", "scene_with_ball.xml"))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Action space: 6 actuators with control range defined in XML
        low = self.model.actuator_ctrlrange[:, 0]
        high = self.model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Observation space: 
        # - Joint positions (6)
        # - Joint velocities (6)
        # - Ball position (3)
        # - Ball velocity (3) 
        # - Site position of gripper (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        self.render_mode = render_mode
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_obs(self):
        # Arm qpos (6)
        arm_qpos = self.data.qpos[:6]
        arm_qvel = self.data.qvel[:6]
        
        # Ball qpos (7) -> we take only px, py, pz
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        ball_qpos_adr = self.model.jnt_qposadr[ball_joint_id]
        ball_pos = self.data.qpos[ball_qpos_adr : ball_qpos_adr + 3].copy()
        
        ball_qvel_adr = self.model.jnt_dofadr[ball_joint_id]
        ball_vel = self.data.qvel[ball_qvel_adr : ball_qvel_adr + 3].copy()
        
        # Gripper site position
        gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        gripper_pos = self.data.site_xpos[gripper_site_id].copy()
        
        return np.concatenate([arm_qpos, arm_qvel, ball_pos, ball_vel, gripper_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        ball_qpos_adr = self.model.jnt_qposadr[ball_joint_id]
        ball_qvel_adr = self.model.jnt_dofadr[ball_joint_id]
        
        # Randomize ball position slightly
        self.data.qpos[ball_qpos_adr] = 0.4 + self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[ball_qpos_adr + 1] = self.np_random.uniform(-0.1, 0.1)
        self.data.qpos[ball_qpos_adr + 2] = 0.05
        
        # Initial velocity towards origin
        self.data.qvel[ball_qvel_adr] = -0.5
        self.data.qvel[ball_qvel_adr + 1] = -self.data.qpos[ball_qpos_adr + 1] * 2.0
        
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.data.ctrl[:] = action
        
        # Frameskip
        frameskip = 5
        for _ in range(frameskip):
            mujoco.mj_step(self.model, self.data)
            
        observation = self._get_obs()
        
        # Reward calculation
        ball_pos = observation[12:15]
        gripper_pos = observation[18:21]
        dist = np.linalg.norm(ball_pos - gripper_pos)
        
        reward = -dist # Basic distance reward
        
        # Bonus for closeness
        if dist < 0.05:
            reward += 2.0
            
        terminated = False
        if ball_pos[2] < -0.05: # Ball fell
            terminated = True
            reward -= 5.0
            
        truncated = False
        info = {"distance": dist}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
