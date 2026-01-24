import mujoco
import mujoco.viewer
import time
import os
import math

def main():
    # Load the model from the XML file
    model_path = os.path.abspath(os.path.join("SO101", "scene_with_ball.xml"))
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    print("ここで環境条件 condition.xmlを追加する") 
    data = mujoco.MjData(model)

    # Initialize renderer for off-screen rendering
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set up output directory
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # Simulation parameters
    duration = 3.0  # total simulation time in seconds
    snapshot_interval = 0.1  # interval between snapshots in seconds
    next_snapshot_time = 0.0

    print(f"Starting simulation. Saving snapshots to '{out_dir}/' every {snapshot_interval}s.")

    while data.time <= duration:
        # Simple sine wave motion for actuators
        t = data.time
        # data.ctrl maps to the actuators defined in the XML
        data.ctrl[0] = 0.5 * math.sin(t)          # shoulder_pan
        data.ctrl[1] = 0.5 * math.cos(t)          # shoulder_lift
        data.ctrl[2] = 0.5 * math.sin(t * 0.5)    # elbow_flex
        data.ctrl[3] = 0.3 * math.cos(t * 1.2)    # wrist_flex
        data.ctrl[4] = 1.0 * math.sin(t * 2.0)    # wrist_roll
        data.ctrl[5] = 0.5 + 0.5 * math.sin(t)    # gripper

        # Check if it's time to take a snapshot
        if data.time >= next_snapshot_time - 1e-6:
            renderer.update_scene(data)
            pixels = renderer.render()
            
            # Use PIL to save the image
            from PIL import Image
            img = Image.fromarray(pixels)
            # Format filename with simulation time
            filename = f"snapshot_{next_snapshot_time:04.1f}s.png"
            img.save(os.path.join(out_dir, filename))
            
            next_snapshot_time += snapshot_interval

        # Step the simulation
        mujoco.mj_step(model, data)

    print(f"Simulation complete. Snapshots saved to '{out_dir}/'.")


if __name__ == "__main__":
    main()
