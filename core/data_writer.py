import csv
import os
import time
import queue
import threading
import logging
import cv2
import numpy as np
import pyzed.sl as sl
from core import shared_state

OUTPUT_DIR = "logs"
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth_frames")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "video")
CSV_PATH = os.path.join(OUTPUT_DIR, "imu_log.csv")
DEPTH_BIN_PATH = os.path.join(OUTPUT_DIR, "depth_stream.bin")  # single append-only file (disabled for now)
VIDEO_PATH_TEMPLATE = os.path.join(VIDEO_DIR, "run_{ts}.mp4")
VIDEO_FPS = 15  # match roughly to expected grab() rate

logger = logging.getLogger("zed_capture")


def setup_output_dirs():
    os.makedirs(DEPTH_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def disk_writer_worker(q, csv_path, video_path, frame_size):
    """
    Writes IMU rows to CSV and the captured BGR frame to an .mp4 video file.

    Depth persistence to disk is disabled for now (see commented block below).
    To re-enable: pass depth_bin_path back in, open it alongside csv/video,
    and uncomment the depth write lines in the loop.
    """
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, frame_size
    )

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "pitch", "yaw", "roll", "frame_index"])
        while True:
            item = q.get()
            if item is None:
                break
            timestamp_ms, pitch, yaw, roll, frame_index, frame_bgr = item

            video_writer.write(frame_bgr)
            writer.writerow([timestamp_ms, pitch, yaw, roll, frame_index])

            # --- depth-to-disk temporarily disabled ---
            # depth_bytes = depth_data.tobytes()
            # depth_bin.write(depth_bytes)
            # offset += len(depth_bytes)

            q.task_done()

    video_writer.release()


def run(zed):
    setup_output_dirs()
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    sensors_data = sl.SensorsData()
    frame_index = 0
    dropped_frames = 0

    write_queue = queue.Queue(maxsize=100)
    writer_thread = None  # started lazily once we know frame size (see below)

    # Preallocated reusable buffers -> avoids per-frame np/cv2 allocation churn.
    # Sizes are set on first frame once we know the real resolution.
    frame_bgr_buf = None
    downsampled_depth_buf = None
    downsampled_depth_f16_buf = None

    last_drop_log = 0.0

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
            timestamp_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()

            bgra_data = image.get_data()
            depth_array = depth.get_data()

            if frame_bgr_buf is None:
                h, w = bgra_data.shape[:2]
                frame_bgr_buf = np.empty((h, w, 3), dtype=np.uint8)
                dh, dw = h // 2, w // 2
                downsampled_depth_buf = np.empty((dh, dw), dtype=np.float32)
                downsampled_depth_f16_buf = np.empty((dh, dw), dtype=np.float16)

                video_path = VIDEO_PATH_TEMPLATE.format(ts=int(time.time()))
                writer_thread = threading.Thread(
                    target=disk_writer_worker,
                    args=(write_queue, CSV_PATH, video_path, (w, h)),
                    daemon=True,
                )
                writer_thread.start()

            # Reuse output buffers via dst= to avoid new allocations every frame
            cv2.cvtColor(bgra_data, cv2.COLOR_BGRA2BGR, dst=frame_bgr_buf)
            cv2.resize(
                depth_array, (0, 0), dst=downsampled_depth_buf,
                fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA,
            )
            # float16 conversion disabled along with depth-to-disk writing (see below)
            # np.copyto(downsampled_depth_f16_buf, downsampled_depth_buf, casting="unsafe")

            imu_pose = sensors_data.get_imu_data().get_pose()
            pitch, yaw, roll = imu_pose.get_euler_angles()

            # queue gets a COPY of the frame, since frame_bgr_buf is reused next iteration
            try:
                write_queue.put_nowait(
                    (timestamp_ms, pitch, yaw, roll, frame_index, frame_bgr_buf.copy())
                )
            except queue.Full:
                dropped_frames += 1
                now = time.monotonic()
                if now - last_drop_log > 1.0:  # rate-limit logging, don't block hot path
                    logger.warning(
                        "Disk yazma hizi yetismiyor, atlanan kare sayisi: %d", dropped_frames
                    )
                    last_drop_log = now

            # --- minimize time spent holding locks: just pointer/scalar assignment ---
            with shared_state.data_lock:
                shared_state.latest_depth_array = downsampled_depth_buf
                shared_state.latest_imu = {"pitch": pitch, "yaw": yaw, "roll": roll}
                shared_state.latest_timestamp = timestamp_ms

            with shared_state.frame_lock:
                shared_state.latest_frame = frame_bgr_buf
                np.copyto(shared_state.shm_rgb, bgra_data)
                np.copyto(shared_state.shm_depth, depth_array)
                shared_state.shm_meta[0] += 1
                shared_state.shm_meta[1] = int(time.time() * 1000)

            shared_state.data_event.set()
            shared_state.frame_event.set()
            frame_index += 1
    finally:
        print("Sistem kapaniyor, kalan veriler diske yaziliyor...")
        write_queue.put(None)
        writer_thread.join()
