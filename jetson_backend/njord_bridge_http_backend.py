#!/usr/bin/env python3
"""
HTTP backend for the Jetson orange_cube_bridge node.

Usage in Njord-Competition bridge/bridge_node.py:

    from jetson_backend.njord_bridge_http_backend import attach_http_backend

    class OrangeCubeBridgeNode(Node):
        def __init__(self):
            ...
            self.create_timer(0.02, self._read_mavlink_messages)
            ...
            attach_http_backend(self, host="0.0.0.0", port=8000)

And inside _read_mavlink_messages(), add these MAVLink messages to the queue:

    elif msg_type in ("MISSION_COUNT", "MISSION_ITEM", "MISSION_ITEM_INT",
                      "MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"):
        self.http_backend_push_mission_message(msg)

This keeps the Pixhawk MAVLink connection owned by the bridge node. The GUI talks
to this HTTP backend, and this backend uses the bridge node's existing master.
"""

import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pymavlink import mavutil


def parse_waypoints_from_text(text):
    waypoints = []
    qgc_wpl = False
    for line_no, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.upper().startswith("QGC WPL"):
            qgc_wpl = True
            continue

        if qgc_wpl:
            waypoint = _parse_qgc_wpl_line(line, len(waypoints) + 1, line_no)
            if waypoint is not None:
                waypoints.append(waypoint)
            continue

        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if len(numbers) < 2:
            continue

        candidates = []
        for index in range(len(numbers) - 1):
            try:
                lat = float(numbers[index])
                lon = float(numbers[index + 1])
            except ValueError:
                continue
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                candidates.append((index, lat, lon))

        if candidates:
            _index, lat, lon = _best_lat_lon_pair(candidates)
            waypoints.append(
                {
                    "name": f"WP_{len(waypoints) + 1:02d}",
                    "lat": lat,
                    "lon": lon,
                    "line": line_no,
                }
            )

    if not waypoints:
        raise ValueError("No valid latitude/longitude waypoint found in TXT content.")
    return waypoints


def _parse_qgc_wpl_line(line, waypoint_index, line_no):
    parts = re.split(r"[\t,; ]+", line.strip())
    if len(parts) < 11:
        return None

    try:
        command = int(float(parts[3]))
        lat = float(parts[8])
        lon = float(parts[9])
        alt = float(parts[10])
    except (TypeError, ValueError, IndexError):
        return None

    if command != mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None

    return {
        "name": f"WP_{waypoint_index:02d}",
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "line": line_no,
    }


def _best_lat_lon_pair(candidates):
    def score(item):
        index, lat, lon = item
        value = index * 0.01
        if 35.0 <= abs(lat) <= 72.0:
            value += 4.0
        if -20.0 <= lon <= 45.0:
            value += 4.0
        if abs(lat) >= abs(lon):
            value += 1.0
        return value

    return max(candidates, key=score)


class MissionBackend:
    def __init__(self, bridge_node):
        self.node = bridge_node
        self._mission_messages = []
        self._mission_lock = threading.Lock()
        self.last_uploaded_waypoints = []
        self.last_mission_name = None

    def push_mission_message(self, msg):
        with self._mission_lock:
            self._mission_messages.append(msg)
            self._mission_messages = self._mission_messages[-80:]

    def _pop_mission_message(self, expected_types, timeout=8.0):
        expected_types = set(expected_types)
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._mission_lock:
                for index, msg in enumerate(self._mission_messages):
                    if msg.get_type() in expected_types:
                        return self._mission_messages.pop(index)
            time.sleep(0.02)
        return None

    def _clear_mission_messages(self):
        with self._mission_lock:
            self._mission_messages.clear()

    def _master(self):
        master = getattr(self.node, "master", None)
        if master is None:
            raise RuntimeError("MAVLink master is not ready.")
        return master

    def _targets(self):
        master = self._master()
        return master.target_system or 1, master.target_component or mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1

    def upload_txt(self, payload):
        mission_name = payload.get("mission_name") or payload.get("filename") or "UPLOADED_TXT"
        waypoints = parse_waypoints_from_text(payload.get("content", ""))
        upload_to_pixhawk = bool(payload.get("upload_to_pixhawk", True))

        if upload_to_pixhawk:
            self.upload_mission(waypoints)
            confirmed_waypoints = self.read_mission()
            if not self._waypoints_match(waypoints, confirmed_waypoints):
                raise RuntimeError("Mission upload verification failed: Pixhawk mission does not match TXT waypoints.")
        else:
            confirmed_waypoints = waypoints

        self.last_uploaded_waypoints = confirmed_waypoints
        self.last_mission_name = mission_name
        return {
            "ok": True,
            "success": True,
            "mission_id": mission_name,
            "mission_name": mission_name,
            "waypoints": confirmed_waypoints,
            "pixhawk_uploaded": upload_to_pixhawk,
            "pixhawk_confirmed": upload_to_pixhawk,
            "message": "Mission TXT parsed and uploaded.",
        }

    def upload_mission(self, waypoints):
        master = self._master()
        target_system, target_component = self._targets()
        self._clear_mission_messages()

        try:
            master.mav.mission_clear_all_send(
                target_system,
                target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            master.mav.mission_clear_all_send(target_system, target_component)

        time.sleep(0.2)
        self._clear_mission_messages()
        self._send_mission_count(target_system, target_component, len(waypoints))

        sent = set()
        deadline = time.time() + max(15.0, len(waypoints) * 3.0)
        while time.time() < deadline:
            msg = self._pop_mission_message(
                ("MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"),
                timeout=2.0,
            )
            if msg is None:
                if not sent:
                    self._send_mission_count(target_system, target_component, len(waypoints))
                continue

            msg_type = msg.get_type()
            if msg_type == "MISSION_ACK":
                result = int(getattr(msg, "type", 0) or 0)
                if len(sent) < len(waypoints):
                    continue
                if result == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    return True
                raise RuntimeError(f"Mission upload rejected: {result}")

            seq = int(getattr(msg, "seq", -1))
            if seq < 0 or seq >= len(waypoints):
                raise RuntimeError(f"Pixhawk requested invalid mission item: {seq}")
            self._send_mission_item(
                target_system,
                target_component,
                seq,
                waypoints[seq],
                use_int=(msg_type == "MISSION_REQUEST_INT"),
            )
            sent.add(seq)

        raise RuntimeError("Mission upload timed out before Pixhawk ACK.")

    def read_mission(self):
        master = self._master()
        target_system, target_component = self._targets()
        self._clear_mission_messages()

        try:
            master.mav.mission_request_list_send(
                target_system,
                target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            master.mav.mission_request_list_send(target_system, target_component)

        count_msg = self._pop_mission_message(("MISSION_COUNT",), timeout=8.0)
        if count_msg is None:
            raise RuntimeError("Pixhawk did not return MISSION_COUNT.")

        count = int(getattr(count_msg, "count", 0) or 0)
        waypoints = []
        for seq in range(count):
            self._request_mission_item(target_system, target_component, seq)
            item = self._pop_mission_message(("MISSION_ITEM_INT", "MISSION_ITEM"), timeout=5.0)
            if item is None:
                raise RuntimeError(f"Pixhawk did not return mission item {seq}.")
            waypoint = self._mission_item_to_waypoint(item, seq)
            if waypoint is not None:
                waypoints.append(waypoint)

        self.last_uploaded_waypoints = waypoints
        return waypoints

    def _waypoints_match(self, expected, actual, tolerance=0.00001):
        if len(expected) != len(actual):
            return False
        for exp, got in zip(expected, actual):
            if abs(float(exp["lat"]) - float(got["lat"])) > tolerance:
                return False
            if abs(float(exp["lon"]) - float(got["lon"])) > tolerance:
                return False
        return True

    def start_mission(self, payload):
        mode = str(payload.get("mode", "AUTO")).upper()
        self.set_mode({"mode": mode})

        master = self._master()
        target_system = master.target_system or 1
        target_component = 0
        master.mav.command_long_send(
            target_system,
            target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        return {
            "ok": True,
            "success": True,
            "mission_id": payload.get("mission_id") or self.last_mission_name,
            "message": "Mission start command sent.",
        }

    def set_arm(self, payload):
        armed = bool(payload.get("armed", True))
        success = self.node._arm_disarm(armed)
        return {
            "ok": bool(success),
            "success": bool(success),
            "message": "ARM command sent." if armed else "DISARM command sent.",
        }

    def set_mode(self, payload):
        mode = str(payload.get("mode") or payload.get("custom_mode") or "AUTO").upper()
        master = self._master()
        mapping = master.mode_mapping()
        if mode not in mapping:
            raise RuntimeError(f"Unknown Pixhawk mode: {mode}")
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mapping[mode],
        )
        return {"ok": True, "success": True, "message": f"Mode command sent: {mode}"}

    def emergency_stop(self):
        self.publish_stop_cmd_vel()
        try:
            self.set_mode({"mode": "HOLD"})
        except Exception:
            pass
        try:
            self.set_arm({"armed": False})
        except Exception:
            pass
        return {"ok": True, "success": True, "message": "Emergency stop sent: stop + HOLD + DISARM."}

    def publish_stop_cmd_vel(self):
        topics = getattr(self.node, "topics", None)
        cmd_vel_pub = getattr(topics, "cmd_vel_pub", None)
        if cmd_vel_pub is None:
            return
        from geometry_msgs.msg import Twist

        msg = Twist()
        for _ in range(10):
            cmd_vel_pub.publish(msg)
            time.sleep(0.02)

    def _send_mission_count(self, target_system, target_component, count):
        master = self._master()
        try:
            master.mav.mission_count_send(
                target_system,
                target_component,
                count,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            master.mav.mission_count_send(target_system, target_component, count)

    def _send_mission_item(self, target_system, target_component, seq, waypoint, use_int=True):
        master = self._master()
        altitude = float(waypoint.get("alt", waypoint.get("altitude", 0.0)) or 0.0)
        if not use_int:
            self._send_mission_item_float(target_system, target_component, seq, waypoint, altitude)
            return

        lat_int = int(float(waypoint["lat"]) * 1e7)
        lon_int = int(float(waypoint["lon"]) * 1e7)
        try:
            master.mav.mission_item_int_send(
                target_system,
                target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                1 if seq == 0 else 0,
                1,
                0,
                0,
                0,
                math.nan,
                lat_int,
                lon_int,
                altitude,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            master.mav.mission_item_int_send(
                target_system,
                target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                1 if seq == 0 else 0,
                1,
                0,
                0,
                0,
                math.nan,
                lat_int,
                lon_int,
                altitude,
            )

    def _send_mission_item_float(self, target_system, target_component, seq, waypoint, altitude):
        master = self._master()
        try:
            master.mav.mission_item_send(
                target_system,
                target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                1 if seq == 0 else 0,
                1,
                0,
                0,
                0,
                math.nan,
                float(waypoint["lat"]),
                float(waypoint["lon"]),
                altitude,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except TypeError:
            master.mav.mission_item_send(
                target_system,
                target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                1 if seq == 0 else 0,
                1,
                0,
                0,
                0,
                math.nan,
                float(waypoint["lat"]),
                float(waypoint["lon"]),
                altitude,
            )

    def _request_mission_item(self, target_system, target_component, seq):
        master = self._master()
        try:
            master.mav.mission_request_int_send(
                target_system,
                target_component,
                seq,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )
        except AttributeError:
            master.mav.mission_request_send(target_system, target_component, seq)
        except TypeError:
            master.mav.mission_request_int_send(target_system, target_component, seq)

    def _mission_item_to_waypoint(self, msg, seq):
        if msg.get_type() == "MISSION_ITEM_INT":
            lat = getattr(msg, "x", 0) / 1e7
            lon = getattr(msg, "y", 0) / 1e7
        else:
            lat = getattr(msg, "x", getattr(msg, "lat", 0.0))
            lon = getattr(msg, "y", getattr(msg, "lon", 0.0))
        command = int(getattr(msg, "command", mavutil.mavlink.MAV_CMD_NAV_WAYPOINT))
        if command != mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
            return None
        if not (-90 <= float(lat) <= 90 and -180 <= float(lon) <= 180):
            return None
        return {"name": f"WP_{seq + 1:02d}", "lat": float(lat), "lon": float(lon)}


class _Handler(BaseHTTPRequestHandler):
    backend = None

    def do_POST(self):
        try:
            payload = self._read_json()
            if self.path == "/api/mission/upload_txt":
                response = self.backend.upload_txt(payload)
            elif self.path == "/api/mission/start":
                response = self.backend.start_mission(payload)
            elif self.path == "/api/mission/current":
                response = {"ok": True, "success": True, "waypoints": self.backend.read_mission()}
            elif self.path == "/api/pixhawk/arm":
                response = self.backend.set_arm(payload)
            elif self.path == "/api/pixhawk/set_mode":
                response = self.backend.set_mode(payload)
            elif self.path == "/api/mission/stop":
                response = self.backend.emergency_stop()
            else:
                self._send_json({"ok": False, "success": False, "message": f"Unknown endpoint: {self.path}"}, 404)
                return
            self._send_json(response)
        except Exception as exc:
            self._send_json({"ok": False, "success": False, "message": str(exc)}, 500)

    def do_GET(self):
        try:
            if self.path == "/api/mission/current":
                self._send_json({"ok": True, "success": True, "waypoints": self.backend.read_mission()})
            elif self.path == "/health":
                self._send_json({"ok": True, "success": True, "message": "NJORD bridge backend is running."})
            else:
                self._send_json({"ok": False, "success": False, "message": f"Unknown endpoint: {self.path}"}, 404)
        except Exception as exc:
            self._send_json({"ok": False, "success": False, "message": str(exc)}, 500)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(body)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        logger = getattr(getattr(self, "backend", None), "node", None)
        if logger is not None:
            logger.get_logger().info("HTTP backend: " + (fmt % args))


def attach_http_backend(bridge_node, host="0.0.0.0", port=8000):
    backend = MissionBackend(bridge_node)
    bridge_node.http_backend = backend
    bridge_node.http_backend_push_mission_message = backend.push_mission_message

    handler = type("NjordBridgeHandler", (_Handler,), {})
    handler.backend = backend
    server = ThreadingHTTPServer((host, int(port)), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="NjordHttpBackend")
    thread.start()
    bridge_node.http_backend_server = server
    bridge_node.get_logger().info(f"NJORD HTTP backend listening on {host}:{port}")
    return backend
