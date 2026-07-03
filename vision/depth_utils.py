"""
test_vision_live.py

Live tests the BuoyDetector on the real ZED2i camera stream.
FPS / CPU / RAM / GPU reporting and depth measurement logic from
the reference script (ZED+YOLO) are adapted to the BuoyDetector.detect() API.

NOTE: This file is NOT a unittest.TestCase – it does not work with the
pytest/unittest runner because it contains an infinite camera loop. Run it
as a separate manual verification script:

    python3 test_vision_live.py --device cuda
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import threading
import time

import cv2
import psutil
import pyzed.sl as sl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # insert(0, ...) is safer than append

from config.camera_config import *
from config.vision_config import BUOY_MODEL_PATH
from vision.detector import BuoyDetector

# ---------------- Settings ----------------
LOG_INTERVAL_SEC = 2.0  # Terminal reporting interval (seconds)
MAX_DEPTH_M = 40.0

# ---- Global Loop Control ----
is_running = True

# ---- GPU Monitoring (background thread, does NOT BLOCK main loop) ----
_gpu_usage_lock = threading.Lock()
_gpu_usage_value = "0% (no data yet)"
_tegrastats_proc = None


def signal_handler(sig, frame):
    """Safely shuts down the program when Ctrl+C (SIGINT) is sent."""
    global is_running
    print("\n[INFO] Shutdown signal received. Exiting...")
    is_running = False


signal.signal(signal.SIGINT, signal_handler)


def _gpu_monitor_thread_nvidia_smi():
    """For desktop/nvidia-smi systems: reads GPU usage periodically."""
    global _gpu_usage_value
    while is_running:
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8', stderr=subprocess.DEVNULL, timeout=1.0
            )
            with _gpu_usage_lock:
                _gpu_usage_value = f"{result.strip()}%"
        except Exception:
            pass
        time.sleep(1.0)


def _gpu_monitor_thread_tegrastats():
    """For Jetson: starts tegrastats ONCE as a continuous process and reads
    its output in the background, writing the last value to _gpu_usage_value.
    The main loop never spawns processes / blocks with readline."""
    global _gpu_usage_value, _tegrastats_proc
    try:
        _tegrastats_proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return

    for line in _tegrastats_proc.stdout:
        if not is_running:
            break
        match = re.search(r'GR3D(?:_FREQ)?\s*(\d+)%', line)
        if match:
            with _gpu_usage_lock:
                _gpu_usage_value = f"{match.group(1)}% (Tegrastats)"


def _gpu_monitor_thread_sysfs():
    """If nvidia-smi and tegrastats are absent: periodic sysfs read (fast, non-blocking)."""
    global _gpu_usage_value
    sysfs_paths = [
        "/sys/class/devfreq/17000000.gv11b/device/load",
        "/sys/class/devfreq/17000000.ga10b/device/load",
        "/sys/devices/gpu.0/load"
    ]
    while is_running:
        for path in sysfs_paths:
            try:
                with open(path, 'r') as f:
                    val = float(f.read().strip())
                    text = f"{val / 10.0:.1f}% (sysfs)" if val > 100 else f"{val:.1f}% (sysfs)"
                    with _gpu_usage_lock:
                        _gpu_usage_value = text
                    break
            except Exception:
                continue
        time.sleep(1.0)


def start_gpu_monitor():
    """Selects the appropriate GPU monitoring method once and starts it in the background (daemon thread)."""
    try:
        subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                encoding='utf-8', stderr=subprocess.DEVNULL, timeout=1.0)
        t = threading.Thread(target=_gpu_monitor_thread_nvidia_smi, daemon=True)
        t.start()
        return
    except Exception:
        pass

    try:
        subprocess.check_output(['which', 'tegrastats'], stderr=subprocess.DEVNULL, timeout=1.0)
        t = threading.Thread(target=_gpu_monitor_thread_tegrastats, daemon=True)
        t.start()
        return
    except Exception:
        pass

    t = threading.Thread(target=_gpu_monitor_thread_sysfs, daemon=True)
    t.start()


def get_gpu_usage() -> str:
    """Instantly returns the last value written by the background thread (non-blocking)."""
    with _gpu_usage_lock:
        return _gpu_usage_value


def parse_args():
    parser = argparse.ArgumentParser(description="BuoyDetector live ZED2i test")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the model on (default: cpu)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without opening a cv2 window, using terminal logs only")
    return parser.parse_args()


# noinspection D
def main():
    global is_running
    args = parse_args()

    start_gpu_monitor()

    print(f"[INFO] Initializing ZED Camera... (Target Resolution reference: {CAMERA_WIDTH}x{CAMERA_HEIGHT})")
    zed = sl.Camera()
    init_params = sl.InitParameters()

    init_params.camera_resolution = CAMERA_RESOLUTION
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = DEPTH_MODE
    init_params.coordinate_units = COORDINATE_UNITS
    init_params.depth_minimum_distance = 0.3

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED: {status}")

    # Break the speed lock with fixed exposure/gain (same logic as ref. script)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 20)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 50)

    runtime_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    print(f"[INFO] Loading BuoyDetector: {BUOY_MODEL_PATH} (device={args.device})")
    detector = BuoyDetector(model_path=BUOY_MODEL_PATH, device=args.device)
    if detector.model is None:
        raise RuntimeError("Failed to load BuoyDetector model.")

    print(f"[INFO] Classes: {detector.class_names}")

    last_log_time = time.time()
    frames_since_last_log = 0

    print(f"\n[INFO] Process started. Status will be reported every {LOG_INTERVAL_SEC} seconds.")
    print("[INFO] Press Ctrl+C in the terminal to stop.\n")

    # CREATING THE WINDOW AS RESIZABLE
    if not args.headless:
        cv2.namedWindow("BuoyDetector - Live Test", cv2.WINDOW_NORMAL)

    try:
        while is_running:
            t_loop_start = time.time()

            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue
            t_grab = time.time()

            current_time = time.time()
            frames_since_last_log += 1

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            t_retrieve = time.time()

            bgra_data = image_zed.get_data()
            frame = cv2.cvtColor(bgra_data, cv2.COLOR_BGRA2BGR)
            depth_map = depth_zed.get_data()
            t_convert = time.time()

            detections = detector.detect(frame, depth_map)
            t_detect = time.time()

            detected_objects = []

            for det in detections:
                label = det.get("class", "?")
                conf = det.get("confidence", 0.0)
                distance = det.get("distance", None)
                bbox = det.get("bbox", None)

                distance_text = f"{distance:.2f}m" if distance is not None else "N/A"
                detected_objects.append(f"{label} ({distance_text})")

                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Class name + confidence score: top left of box
                    label_text = f"{label} {conf:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Depth info: top right of box (measure width to align right)
                    (text_w, _), _ = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    depth_x = max(0, x2 - text_w)
                    cv2.putText(frame, distance_text, (depth_x, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            t_draw = time.time()

            elapsed_time = current_time - last_log_time
            if elapsed_time >= LOG_INTERVAL_SEC:
                fps = frames_since_last_log / elapsed_time
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                gpu_info = get_gpu_usage()

                print(f"--- [ {time.strftime('%H:%M:%S')} System Status ] ---")
                print(f"FPS  : {fps:.1f}")
                print(f"CPU  : {cpu_usage:.1f}%")
                print(f"RAM  : {ram_usage:.1f}%")
                print(f"GPU  : {gpu_info}")
                print(f"Obj  : {', '.join(detected_objects) if detected_objects else 'None'}")
                print("-" * 35)

                last_log_time = current_time
                frames_since_last_log = 0

            if not args.headless:
                cv2.imshow("BuoyDetector - Live Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    is_running = False
            t_show = time.time()

            # --- TIMING DIAGNOSTICS: Catch slow frames ---
            step_grab = t_grab - t_loop_start
            step_retrieve = t_retrieve - t_grab
            step_convert = t_convert - t_retrieve
            step_detect = t_detect - t_convert
            step_draw = t_draw - t_detect
            step_show = t_show - t_draw
            step_total = t_show - t_loop_start

            if step_total > 0.15:  # Over 150ms = noticeable stutter
                print(f"[SLOW FRAME] TOTAL={step_total * 1000:.0f}ms | "
                      f"grab={step_grab * 1000:.0f}ms retrieve={step_retrieve * 1000:.0f}ms "
                      f"convert={step_convert * 1000:.0f}ms detect={step_detect * 1000:.0f}ms "
                      f"draw={step_draw * 1000:.0f}ms show={step_show * 1000:.0f}ms")

    finally:
        print("[INFO] Releasing resources...")
        is_running = False
        if _tegrastats_proc is not None:
            try:
                _tegrastats_proc.terminate()
            except Exception:
                pass
        zed.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Shutdown successful.")


if __name__ == "__main__":
    main()
