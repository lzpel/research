import gymnasium as gym
from catch_env import CatchEnv
import numpy as np
import os
from PIL import Image

def main():
    # Instantiate the environment
    env = CatchEnv(render_mode="rgb_array")
    
    # Reset the environment
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial distance to ball:", info.get("distance"))
    
    # Create out directory for verification snapshots
    os.makedirs("out_rl", exist_ok=True)
    
    # Run a few steps with random actions
    for i in range(20):
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render and save every 5 steps
        if i % 5 == 0:
            frame = env.render()
            if frame is not None:
                img = Image.fromarray(frame)
                img.save(f"out/step_{i:02d}.png")
        
        print(f"Step {i:02d}: Reward = {reward:.4f}, Dist = {info.get('distance'):.4f}")
        
        if terminated or truncated:
            print("Episode finished")
            break
            
    env.close()
    print("Environment test complete. Snapshots saved to 'out_rl/'.")

if __name__ == "__main__":
    main()
