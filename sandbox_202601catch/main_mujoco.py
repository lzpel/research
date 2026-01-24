import mujoco
import mujoco.viewer
import time
import os

def main():
    # Load the model from the XML file
    model_path = os.path.join("SO101", "scene.xml")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Use the passive viewer for visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo Viewer started. Close the window to exit.")
        while viewer.is_running():
            step_start = time.time()

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
