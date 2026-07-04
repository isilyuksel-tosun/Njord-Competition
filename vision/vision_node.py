import json
import time
from multiprocessing import shared_memory

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from config.camera_config import RGB_SHAPE, DEPTH_SHAPE
from config.vision_config import BUOY_MODEL_PATH, VESSEL_MODEL_PATH
from vision.detector import BuoyDetector, VesselDetector

TASK_DETECTOR_MAP = {
    "task1": {"buoy"},
    "task2": {"buoy", "vessel"},
    "task3": {"vessel"},
}

DETECTOR_REGISTRY = {
    "buoy": (BuoyDetector, BUOY_MODEL_PATH),
    "vessel": (VesselDetector, VESSEL_MODEL_PATH),
}


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        self.detectors = {}  # name -> instance
        self.current_task = None

        self.rgb_shm = self._attach_with_retry("RGB_DATA")
        self.depth_shm = self._attach_with_retry("DEPTH_DATA")
        self.meta_shm = self._attach_with_retry("ZED_META")
        self.rgb = np.ndarray(RGB_SHAPE, dtype=np.uint8, buffer=self.rgb_shm.buf)
        self.depth = np.ndarray(DEPTH_SHAPE, dtype=np.float32, buffer=self.depth_shm.buf)
        self.meta = np.ndarray((2,), dtype=np.int64, buffer=self.meta_shm.buf)

        self.last_frame_id = -1
        self.pub = self.create_publisher(String, '/vision/detections', 10)

        self.create_subscription(String, '/mission/active_task', self.on_task_change, 10)

        self.create_timer(1 / 15, self.process_frame)

    def on_task_change(self, msg: String):
        task = msg.data
        if task == self.current_task:
            return

        wanted = TASK_DETECTOR_MAP.get(task)
        if wanted is None:
            self.get_logger().warn(f"Unknown task: '{task}', detector state unchanged.")
            return

        self.current_task = task
        self.get_logger().info(f"Task changed -> '{task}', active detectors: {wanted}")

        for name in list(self.detectors.keys()):
            if name not in wanted:
                self.get_logger().info(f"Closing '{name}' detector...")
                del self.detectors[name]

        for name in wanted:
            if name not in self.detectors:
                cls, model_path = DETECTOR_REGISTRY[name]
                self.get_logger().info(f"Loading '{name}' detector...")
                self.detectors[name] = cls(model_path=model_path)

        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def _attach_with_retry(self, name, retries=20, delay=0.5):
        for _ in range(retries):
            try:
                return shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                time.sleep(delay)
        raise RuntimeError(f"{name} shared memory not found")

    def process_frame(self):
        frame_id = int(self.meta[0])
        if frame_id == self.last_frame_id:
            return
        self.last_frame_id = frame_id

        if not self.detectors:
            return

        bgr_image = self.rgb[:, :, :3].copy()
        depth_array = self.depth.copy()

        all_detections = []
        for name, detector in self.detectors.items():
            dets = detector.detect(bgr_image, depth_array)
            for d in dets:
                d["type"] = name
            all_detections += dets

        if all_detections:
            msg = String()
            msg.data = json.dumps({
                "frame_id": frame_id,
                "detections": all_detections,
            })
            self.pub.publish(msg)


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
