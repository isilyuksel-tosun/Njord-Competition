import threading
from multiprocessing import shared_memory

import numpy as np

from config.camera_config import CAMERA_WIDTH, CAMERA_HEIGHT

latest_frame = None
frame_lock = threading.Lock()
frame_event = threading.Event()

latest_depth_array = None  # np array
latest_imu = None  # pitch, yaw, roll
latest_timestamp = None  # ms
data_lock = threading.Lock()
data_event = threading.Event()

RGB_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, 4)
DEPTH_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH)

# Create Shared Memory for RGB, Depth, and Metadata
_rgb_shm = shared_memory.SharedMemory(name="RGB_DATA", create=True, size=int(np.prod(RGB_SHAPE)))
_depth_shm = shared_memory.SharedMemory(name="DEPTH_DATA", create=True, size=int(np.prod(DEPTH_SHAPE) * 4))
_meta_shm = shared_memory.SharedMemory(name="ZED_META", create=True, size=16)  # [frame_id, timestamp_ms]

shm_rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=_rgb_shm.buf)
shm_depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=_depth_shm.buf)
shm_meta = np.ndarray((2,), dtype=np.int64, buffer=_meta_shm.buf)
