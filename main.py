import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from multiprocessing import get_context

from core import capture_proc
from core import data_writer
from servers import data_server
from servers import video_server

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def start_capture_process():
    mp_context = get_context("spawn")
    frame_lock = mp_context.Lock()
    frame_ready_event = mp_context.Event()
    stop_event = mp_context.Event()
    ready_queue = mp_context.Queue(maxsize=1)

    process = mp_context.Process(
        target=capture_proc.run_capture,
        kwargs={
            "lock": frame_lock,
            "frame_ready_event": frame_ready_event,
            "stop_event": stop_event,
            "ready_queue": ready_queue,
        },
        daemon=False,
    )

    print("[SYSTEM] ZED capture process is starting with spawn context...")
    process.start()

    try:
        ready_msg = ready_queue.get(timeout=20)
    except queue.Empty as exc:
        stop_event.set()
        process.terminate()
        process.join(timeout=2)
        raise RuntimeError("ZED capture process did not become ready in time.") from exc

    if "error" in ready_msg:
        stop_event.set()
        process.join(timeout=2)
        raise RuntimeError(f"ZED capture process failed: {ready_msg['error']}")

    fx = ready_msg["fx"]
    cx = ready_msg["cx"]
    print(f"[SYSTEM] ZED calibration loaded: fx={fx:.2f}, cx={cx:.2f}")

    return process, frame_lock, frame_ready_event, stop_event, fx, cx


if __name__ == "__main__":
    fx = None
    cx = None
    capture_process = None
    capture_stop_event = None
    frame_lock = None
    frame_ready_event = None
    child_processes = []

    try:
        (
            capture_process,
            frame_lock,
            frame_ready_event,
            capture_stop_event,
            fx,
            cx,
        ) = start_capture_process()

        # Flask
        threading.Thread(target=video_server.start, args=(5000,), daemon=True).start()
        threading.Thread(target=data_server.start, args=(5001,), daemon=True).start()

        print("[SYSTEM] ZED capture was launched with success.")
        print("[SYSTEM] Video stream   -> http://0.0.0.0:5000/data/stream")
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

        vision_args_setup = f"--fx {shlex.quote(str(fx))} --cx {shlex.quote(str(cx))}"

        ################################################################################################################
        # SETUP NJORD MISSION PATHS
        ################################################################################################################
        njord_task1_path = os.path.join(PROJECT_ROOT, "missions", "task1_maneuvering_and_path_finding.py")
        njord_task2_path = os.path.join(PROJECT_ROOT, "missions", "task2_collision_avoidance.py")
        njord_task3_path = os.path.join(PROJECT_ROOT, "missions", "task3_docking.py")

        ################################################################################################################

        cmd_vision = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(vision_path)} {vision_args_setup}"
        )
        cmd_bridge = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(bridge_path)}"
        )
        ################################################################################################################
        # SETUP NJORD MISSION COMMANDS
        ################################################################################################################
        cmd_njord_task1 = (
            f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(njord_task1_path)}"
        )
        # cmd_njord_task2 = (
        #     f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(njord_task2_path)}"
        # )
        # cmd_njord_task3 = (
        #     f"{ros2_setup} && {python_path_setup} && {shlex.quote(sys.executable)} {shlex.quote(njord_task3_path)}"
        # )
        ################################################################################################################

        p_bridge = subprocess.Popen(cmd_bridge, shell=True, executable="/bin/bash")
        child_processes.append(p_bridge)
        print(f" -> Bridge Node launched (PID: {p_bridge.pid})")

        p_vision = subprocess.Popen(cmd_vision, shell=True, executable="/bin/bash")
        child_processes.append(p_vision)
        print(f" -> Vision Node launched (PID: {p_vision.pid})")

        time.sleep(2)

        ################################################################################################################
        #   NJORD MISSION START CMD
        ################################################################################################################
        p_njord_task1 = subprocess.Popen(cmd_njord_task1, shell=True, executable="/bin/bash")
        child_processes.append(p_njord_task1)
        print(f" -> NJORD Mission 1 Node launched (PID: {p_njord_task1.pid})\n")

        # p_njord_task2 = subprocess.Popen(cmd_njord_task2, shell=True, executable="/bin/bash")
        # child_processes.append(p_njord_task2)
        # print(f" -> NJORD Mission 2 Node launched (PID: {p_njord_task2.pid})\n")
        #
        # p_njord_task3 = subprocess.Popen(cmd_njord_task3, shell=True, executable="/bin/bash")
        # child_processes.append(p_njord_task3)
        # print(f" -> NJORD Mission 3 Node launched (PID: {p_njord_task3.pid})\n")
        ################################################################################################################

        print("[SYSTEM] System active. Ctrl+C at the terminal to close.")

        data_writer.run(frame_lock, frame_ready_event, capture_stop_event)

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

        if capture_stop_event is not None:
            capture_stop_event.set()

        if capture_process is not None:
            capture_process.join(timeout=3)
            if capture_process.is_alive():
                capture_process.terminate()
                capture_process.join(timeout=2)
            print("[SYSTEM] ZED capture process closed.")

        print("[SYSTEM] The entire system was safely stopped.")
