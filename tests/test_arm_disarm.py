#!/usr/bin/env python3

import argparse
import os
import time

from pymavlink import mavutil

DEFAULT_CONNECTION_STRING = os.getenv("MAVLINK_CONNECTION_STRING", "/dev/ttyACM0")
DEFAULT_BAUD = int(os.getenv("MAVLINK_BAUD", "115200"))
DEFAULT_HEARTBEAT_TIMEOUT = int(os.getenv("MAVLINK_HEARTBEAT_TIMEOUT", "15"))
DEFAULT_SECONDS = 5.0
DEFAULT_STEERING_PWM = 1500
DEFAULT_THROTTLE_PWM = 1650
NEUTRAL_PWM = 1500
RC_PERIOD_SEC = 0.05


def clamp_pwm(value):
    return max(1100, min(1900, int(value)))


def connect_mavlink(connection_string, baud, heartbeat_timeout):
    print(f"[TEST] Connecting MAVLink: {connection_string}, baud={baud}")
    master = mavutil.mavlink_connection(connection_string, baud=baud)
    print("[TEST] Waiting heartbeat...")
    master.wait_heartbeat(timeout=heartbeat_timeout)
    print(
        f"[TEST] Heartbeat received: target_system={master.target_system}, "
        f"target_component={master.target_component}"
    )
    return master


def get_targets(master, target_system_arg, target_component_arg):
    target_system = target_system_arg or master.target_system or 1
    target_component = target_component_arg or master.target_component or 1

    if target_system == 0:
        target_system = 1
    if target_component == 0:
        target_component = 1

    print(f"[TEST] Using target_system={target_system}, target_component={target_component}")
    return target_system, target_component


def set_mode(master, target_system, mode_name):
    mapping = master.mode_mapping()
    if mode_name not in mapping:
        print(f"[TEST] Mode {mode_name!r} not available. Available modes: {sorted(mapping)}")
        return False

    master.mav.set_mode_send(
        target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mapping[mode_name],
    )
    print(f"[TEST] {mode_name} mode command sent.")
    return True


def send_arm_disarm(master, target_system, target_component, arm):
    master.mav.command_long_send(
        target_system,
        target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1 if arm else 0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    print("[TEST] ARM command sent." if arm else "[TEST] DISARM command sent.")


def wait_for_armed(master, expected_armed, timeout_sec=8.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
        if msg is None:
            continue

        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        mode = mavutil.mode_string_v10(msg)
        print(f"[TEST] Heartbeat state: armed={armed}, mode={mode}")
        if armed == expected_armed:
            return True

    return False


def send_rc_override(master, target_system, target_component, steering_pwm, throttle_pwm, throttle_channel):
    channels = [0, 0, 0, 0, 0, 0, 0, 0]
    channels[0] = clamp_pwm(steering_pwm)
    channels[throttle_channel - 1] = clamp_pwm(throttle_pwm)

    master.mav.rc_channels_override_send(
        target_system,
        target_component,
        channels[0],
        channels[1],
        channels[2],
        channels[3],
        channels[4],
        channels[5],
        channels[6],
        channels[7],
    )


def send_manual_control(master, target_system, steering_pwm, throttle_pwm):
    steering = int((clamp_pwm(steering_pwm) - 1500) * (1000 / 400))
    throttle = int((clamp_pwm(throttle_pwm) - 1100) * (1000 / 800))
    steering = max(-1000, min(1000, steering))
    throttle = max(0, min(1000, throttle))

    master.mav.manual_control_send(
        target_system,
        0,
        steering,
        throttle,
        0,
        0,
    )


def stop_vehicle(master, target_system, target_component, throttle_channel, seconds=1.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        send_rc_override(
            master,
            target_system,
            target_component,
            NEUTRAL_PWM,
            NEUTRAL_PWM,
            throttle_channel,
        )
        send_manual_control(master, target_system, NEUTRAL_PWM, NEUTRAL_PWM)
        time.sleep(RC_PERIOD_SEC)


def release_rc_override(master, target_system, target_component):
    master.mav.rc_channels_override_send(
        target_system,
        target_component,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def run_motors(master, target_system, target_component, args):
    steering_pwm = clamp_pwm(args.steering_pwm)
    throttle_pwm = clamp_pwm(args.throttle_pwm)
    deadline = time.monotonic() + args.seconds

    print(
        f"[TEST] Running for {args.seconds:.1f}s: "
        f"steering_ch1={steering_pwm}, throttle_ch{args.throttle_channel}={throttle_pwm}"
    )
    print("[TEST] Sending both RC_CHANNELS_OVERRIDE and MANUAL_CONTROL.")

    while time.monotonic() < deadline:
        send_rc_override(
            master,
            target_system,
            target_component,
            steering_pwm,
            throttle_pwm,
            args.throttle_channel,
        )
        send_manual_control(master, target_system, steering_pwm, throttle_pwm)
        time.sleep(RC_PERIOD_SEC)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arm, run motors for 5 seconds, then disarm using MAVLink."
    )
    parser.add_argument("--connection-string", default=DEFAULT_CONNECTION_STRING)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--heartbeat-timeout", type=int, default=DEFAULT_HEARTBEAT_TIMEOUT)
    parser.add_argument("--target-system", type=int, default=None)
    parser.add_argument("--target-component", type=int, default=None)
    parser.add_argument("--mode", default="MANUAL")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--steering-pwm", type=int, default=DEFAULT_STEERING_PWM)
    parser.add_argument("--throttle-pwm", type=int, default=DEFAULT_THROTTLE_PWM)
    parser.add_argument("--throttle-channel", type=int, choices=range(1, 9), default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    armed_by_test = False

    print("[TEST] self-contained arm/disarm motor test started.")
    master = connect_mavlink(args.connection_string, args.baud, args.heartbeat_timeout)
    target_system, target_component = get_targets(master, args.target_system, args.target_component)

    try:
        set_mode(master, target_system, args.mode)
        time.sleep(1.0)

        send_arm_disarm(master, target_system, target_component, True)
        armed_by_test = True

        if not wait_for_armed(master, True):
            raise RuntimeError("Vehicle did not report armed=True after ARM command.")

        run_motors(master, target_system, target_component, args)

    except KeyboardInterrupt:
        print("[TEST] Interrupted by user.")

    finally:
        print("[TEST] Stopping vehicle...")
        stop_vehicle(master, target_system, target_component, args.throttle_channel)

        if armed_by_test:
            send_arm_disarm(master, target_system, target_component, False)
            wait_for_armed(master, False)
            stop_vehicle(master, target_system, target_component, args.throttle_channel, seconds=0.5)

        release_rc_override(master, target_system, target_component)
        print("[TEST] Done.")


if __name__ == "__main__":
    main()
