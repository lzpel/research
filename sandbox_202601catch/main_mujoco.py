import mujoco
import mujoco.viewer
import time
import os
import math

def main():
    # Load the model from the XML file
    model_path = os.path.join("SO101", "scene.xml")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    print("ここで環境条件 condition.xmlを追加する") 
    data = mujoco.MjData(model)

    # Use the passive viewer for visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo Viewer started. Close the window to exit.")
        while viewer.is_running():
            step_start = time.time()

            # Simple sine wave motion for actuators
            t = data.time
            # data.ctrl maps to the actuators defined in the XML
            # Let's move them slightly
            data.ctrl[0] = 0.5 * math.sin(t)          # shoulder_pan
            data.ctrl[1] = 0.5 * math.cos(t)          # shoulder_lift
            data.ctrl[2] = 0.5 * math.sin(t * 0.5)    # elbow_flex
            data.ctrl[3] = 0.3 * math.cos(t * 1.2)    # wrist_flex
            data.ctrl[4] = 1.0 * math.sin(t * 2.0)    # wrist_roll
            data.ctrl[5] = 0.5 + 0.5 * math.sin(t)    # gripper

            # Step the simulation
            mujoco.mj_step(model, data)

            # Pick up changes from the viewer and sync back
            viewer.sync()

            # Rudimentary time synchronization
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
