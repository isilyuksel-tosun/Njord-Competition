import os
import shlex
import subprocess
import sys
import threading
import time

from config.camera_config import *
from core import data_writer
from core import shared_state
from servers import data_server
from servers import video_server

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def init_camera():
    zed = sl.Camera()
    init = sl.InitParameters()

    init.depth_mode = DEPTH_MODE
    init.coordinate_units = COORDINATE_UNITS
    init.camera_resolution = CAMERA_RESOLUTION
    init.camera_fps = CAMERA_FPS

    print("[SYSTEM] ZED Camera is starting...")
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED could not open. Check the camera connection.")
    return zed


if __name__ == "__main__":
    zed = None
    child_processes = []

    try:
        zed = init_camera()
        # Flask
        threading.Thread(target=video_server.start, args=(5000,), daemon=True).start()
        threading.Thread(target=data_server.start, args=(5001,), daemon=True).start()

        print("[SYSTEM] ZED was launched with success.")
        print("[SYSTEM] Video stream  -> http://0.0.0.0:5000/video_feed")
        print("[SYSTEM] Data stream   -> http://0.0.0.0:5001/data/stream")

        print("\n[SYSTEM] Vision and bridge node launch in ROS2...")
        time.sleep(1)

        if os.path.isfile("/opt/ros/kilted/setup.bash"):
            ros2_setup = "source /opt/ros/kilted/setup.bash"
        else:
            ros2_setup = "source /opt/ros/foxy/setup.bash"

        python_path_setup = f"export PYTHONPATH={shlex.quote(PROJECT_ROOT)}:$PYTHONPATH"

        vision_path = os.path.join(PROJECT_ROOT, "vision", "vision_node.py")
        bridge_path = os.path.join(PROJECT_ROOT, "bridge", "bridge_node.py")

        task1_path = os.path.join(PROJECT_ROOT, "missions", "task1_maneuvering_and_path_finding.py")

        cmd_vision = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(vision_path)}"
        )
        cmd_bridge = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(bridge_path)}"
        )

        # ------------------------------
        #   TASK START CMD
        # ------------------------------
        cmd_task1 = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(task1_path)}"
        )
        # ------------------------------

        p_bridge = subprocess.Popen(cmd_bridge, shell=True, executable="/bin/bash")
        child_processes.append(p_bridge)
        print(f" -> Bridge Node launched (PID: {p_bridge.pid})")

        p_vision = subprocess.Popen(cmd_vision, shell=True, executable="/bin/bash")
        child_processes.append(p_vision)
        print(f" -> Vision Node launched (PID: {p_vision.pid})")

        time.sleep(2)

        # ------------------------------
        #   TASK START PROCESS
        # ------------------------------

        p_task1 = subprocess.Popen(cmd_task1, shell=True, executable="/bin/bash")
        child_processes.append(p_task1)
        print(f" -> Mission 1 Node launched (PID: {p_task1.pid})\n")
        # ------------------------------

        print("[SYSTEM] System active. Ctrl+C at the terminal to close.")

        data_writer.run(zed)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopped by the user (Ctrl+C)...")
    except Exception as exc:
        print(f"[SYSTEM] Hata olustu: {exc}")
        raise
    finally:
        print("[SYSTEM] Cleaning process was started...")

        for p in child_processes:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception as exc:
                print(f"[SYSTEM] Error while sub-process shut down: {exc}")

        print("[SYSTEM] Sub-processes closed.")

        if zed is not None:
            zed.close()
            print("[SYSTEM] ZED closed.")

        try:
            shared_state._rgb_shm.close()
            shared_state._rgb_shm.unlink()
            shared_state._depth_shm.close()
            shared_state._depth_shm.unlink()
            shared_state._meta_shm.close()
            shared_state._meta_shm.unlink()
            print("[SYSTEM] Shared memory cleared.")
        except Exception:
            pass

        print("[SYSTEM] The entire system was safely stopped.")
