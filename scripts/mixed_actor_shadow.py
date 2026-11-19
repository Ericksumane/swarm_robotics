"""Run the frozen mixed-platform actors against ROS odometry, with shadow output."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from check_rosbot_formation_path import check
from cooperative_pose_reference import (
    CooperativePoseReference, CooperativeTranslationReference, hold_pose_ready, stopped,
)
from cooperative_telemetry import CooperativeTelemetry
from swarm_core.mixed_commands import constrain_mixed_commands


ROOT = Path(__file__).resolve().parents[1]
ZERO = [[0., 0.] for _ in range(3)]


def read(path):
    return json.loads(path.read_text())


def load_actor(directory, hashes, name, contract, inputs, outputs):
    path = directory / name
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != hashes[name]:
        raise ValueError(f"Checksum mismatch for {name}.")
    artifact = json.loads(content)
    if artifact["contract"] != contract:
        raise ValueError(f"Unexpected actor contract in {name}.")
    weights = artifact["weights"]
    first, bias, second, output = [
        np.asarray(weights[key], dtype=np.float32)
        for key in ("0.weight", "0.bias", "2.weight", "2.bias")
    ]
    if ((first.shape, bias.shape, second.shape, output.shape)
            != ((32, inputs), (32,), (outputs, 32), (outputs,))
            or not all(np.isfinite(array).all() for array in (first, bias, second, output))):
        raise ValueError(f"Invalid weights in {name}.")

    def infer(features):
        return second @ np.tanh(first @ np.asarray(features, dtype=np.float32) + bias) + output

    return infer


class MixedActorShadow(Node):
    def __init__(self, profile=None):
        robot_id = None if profile is None else profile["robot_id"]
        super().__init__("mixed_actor_shadow" if robot_id is None else f"mixed_actor_shadow_{robot_id}")
        self.robot_id = robot_id
        self.declare_parameter("common_frame", "mission")
        self.declare_parameter("max_odom_age_s", .30)
        self.frame = self.get_parameter("common_frame").value
        self.max_age = self.get_parameter("max_odom_age_s").value
        if not self.frame or not math.isfinite(self.max_age) or not 0. < self.max_age <= 1.:
            raise ValueError("Invalid odometry frame or age limit.")
        manifest = read(ROOT / "experiments/mixed_campaign_160k_candidate.json")
        directory = ROOT / manifest["actor_directory"]
        hashes = manifest["sha256"]
        self.anchor_actors = {
            index: load_actor(directory, hashes, f"{robot}_actor.json", "anchor_pose_v1", 5, 2)
            for index, robot in ((0, "go2"), (2, "jackal"))
        }
        self.wheeltec_actor = load_actor(directory, hashes, "wheeltec_actor.json",
                                        "wheeltec_vertex_arc_v1", 10, 2)
        self.speed_actor = load_actor(directory, hashes, "translation_actor.json",
                                     "translation_speed_v1", 3, 1)
        self.task = read(ROOT / "configs/tasks/mixed_formation_nominal080.json")
        self.samples = {}
        ids = range(3) if robot_id is None else (robot_id,)
        prefix = "/swarm/shadow" if profile is None else f"/swarm/{profile['session']}/shadow"
        self.commands = {
            i: self.create_publisher(Twist, f"{prefix}/robot_{i}/cmd_vel", 10)
            for i in ids
        }
        status_topic = f"{prefix}/status" if profile is None else f"{prefix}/robot_{robot_id}/status"
        self.status_pub = self.create_publisher(String, status_topic, 10)
        state_topics = [
            f"/robot_{i}/odom" if profile is None else f"{prefix}/robot_{i}/odom"
            for i in range(3)
        ]
        self.subscriptions_by_robot = [
            self.create_subscription(Odometry, topic,
                                     lambda msg, i=i: self.odometry(i, msg),
                                     qos_profile_sensor_data)
            for i, topic in enumerate(state_topics)
        ]
        if profile is not None:
            self.profile = profile
            self.transform = None
            self.local_stamp = None
            self.state_pub = self.create_publisher(
                Odometry, state_topics[robot_id], qos_profile_sensor_data)
            self.raw_subscription = self.create_subscription(
                Odometry, profile["odom_topic"], self.local_odometry, qos_profile_sensor_data)
        self.plan_id = None
        self.phase = "waiting_for_odometry"
        self.fault = None
        self.pose_controller = None
        self.translation = None
        self.motion_started = False
        self.hold = 0
        self.started = None
        self.timer = self.create_timer(.1, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def local_odometry(self, msg):
        pose = msg.pose.pose
        q = pose.orientation
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds / 1e9
        if not -.05 <= now - stamp <= self.max_age:
            return
        values = (pose.position.x, pose.position.y, q.x, q.y, q.z, q.w,
                  msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        if (msg.header.frame_id != self.profile["odom_frame"]
                or msg.child_frame_id != self.profile["body_frame"]
                or not all(math.isfinite(v) for v in values)
                or abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1.) >= .01):
            if self.started is not None:
                self.fault = "invalid_local_odometry"
            return
        if self.local_stamp is not None and stamp <= self.local_stamp:
            if stamp < self.local_stamp and self.started is not None:
                self.fault = "local_odometry_clock_reset"
            return
        self.local_stamp = stamp
        x, y = pose.position.x, pose.position.y
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        if self.transform is None:
            target_x, target_y, target_yaw_deg = self.profile["initial_pose"]
            angle = math.radians(target_yaw_deg) - yaw
            c, s = math.cos(angle), math.sin(angle)
            self.transform = c, s, target_x - c*x + s*y, target_y - s*x - c*y, angle
        c, s, tx, ty, angle = self.transform
        state = Odometry()
        state.header.stamp = msg.header.stamp
        state.header.frame_id = self.frame
        state.child_frame_id = f"robot_{self.robot_id}/base_link"
        state.pose.pose.position.x = tx + c*x - s*y
        state.pose.pose.position.y = ty + s*x + c*y
        state.pose.pose.orientation.z = math.sin((yaw + angle)/2)
        state.pose.pose.orientation.w = math.cos((yaw + angle)/2)
        state.twist = msg.twist
        self.state_pub.publish(state)

    def odometry(self, index, msg):
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        twist = msg.twist.twist
        values = (position.x, position.y, q.x, q.y, q.z, q.w, twist.linear.x, twist.angular.z)
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        valid = (msg.header.frame_id == self.frame
                 and msg.child_frame_id == f"robot_{index}/base_link"
                 and all(math.isfinite(value) for value in values)
                 and abs(sum(value**2 for value in (q.x, q.y, q.z, q.w)) - 1.) < .01)
        if not valid:
            self.samples.pop(index, None)
            if self.started is not None:
                self.fault = f"invalid_odometry_robot_{index}"
            return
        previous = self.samples.get(index)
        if previous is not None and stamp <= previous[0]:
            return
        heading = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        self.samples[index] = (stamp, time.monotonic(), dict(
            x=position.x, y=position.y, heading=heading,
            speed=twist.linear.x, yaw_rate=twist.angular.z))

    def pose_feedback(self, goals, precision=False):
        controller = CooperativePoseReference(
            goals, self.task["operating_area"], precision=precision,
            wheeltec_heading_tolerance_rad=math.radians(3),
            wheeltec_reversal_deceleration_mps2=.30,
            wheeltec_goal_pair_tolerance_m=.05 if precision else None,
        )
        controller.anchor_pose_actors = self.anchor_actors
        controller.wheeltec_actor = self.wheeltec_actor
        controller.wheeltec_actor_uses_vertex = True
        return controller

    def plan(self, states):
        poses = [(s["x"], s["y"], s["heading"]) for s in states]
        if self.robot_id is not None:
            poses = [tuple(round(value, 4) + 0. for value in pose) for pose in poses]
        plan = check(0, poses, self.task, choose_reachable=True)
        if not plan["found"]:
            self.fault = "formation_plan_unresolved"
            return
        goals = [(*plan["staged_anchors"][0], 0.), (*plan["goal"], 0.),
                 (*plan["staged_anchors"][1], 0.)]
        plan_bytes = json.dumps([goals, plan["primitives"], plan["start"]],
                                sort_keys=True, allow_nan=False).encode()
        self.plan_id = hashlib.sha256(plan_bytes).hexdigest()
        self.pose_controller = self.pose_feedback(goals)
        self.pose_controller.path = list(plan["primitives"])
        self.pose_controller.segment_start = tuple(plan["start"])
        self.phase = "formation"
        self.started = time.monotonic()

    def control(self, states):
        distance = min(math.hypot(a["x"]-b["x"], a["y"]-b["y"])
                       for i, a in enumerate(states) for b in states[i+1:])
        if distance < .65:
            self.fault = "center_clearance"
            return ZERO
        if self.started is None:
            self.plan(states)
            if self.fault:
                return ZERO
        if time.monotonic() - self.started >= 165.:
            self.fault = "mission_timeout"
            return ZERO
        if self.phase == "translation":
            commands = self.translation.command(states)
            near_end = self.translation.reference_progress >= self.translation.distance_m - .08
            if near_end and all(stopped(state) for state in states):
                self.pose_controller = self.pose_feedback(
                    [(*goal, 0.) for goal in self.translation.goals], precision=True)
                self.phase = "terminal"
            return commands
        if self.phase == "complete":
            return ZERO
        if self.phase == "terminal" and hold_pose_ready(states, self.pose_controller.goals,
                                                       pair_tolerance_m=.05):
            self.hold = self.hold + 1 if all(stopped(state) for state in states) else 0
            if self.hold >= 20:
                self.phase = "complete"
            return ZERO
        self.hold = 0
        commands, status = self.pose_controller.command(states)
        if status == "unresolved":
            self.fault = "pose_unresolved"
            return ZERO
        if self.phase == "formation" and status == "settled":
            self.translation = CooperativeTranslationReference(
                states, group_speed_mps=.32, adaptive_speed=True,
                smooth_entry=True, terminal_recovery=True)
            self.translation.speed_actor = lambda features, _: float(self.speed_actor(features)[0])
            self.phase = "translation"
            self.motion_started = True
            return ZERO
        return commands

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        wall = time.monotonic()
        commands = ZERO
        fresh = len(self.samples) == 3 and all(
            -.05 <= now-stamp <= self.max_age and wall-received <= self.max_age
            for stamp, received, _ in self.samples.values())
        if self.started is not None and not fresh:
            self.fault = self.fault or "odometry_stale"
        if fresh and not self.fault:
            states = [self.samples[i][2] for i in range(3)]
            try:
                commands = constrain_mixed_commands(self.control(states))
                commands = CooperativeTelemetry.filter_local_clearance(states, commands)
                if self.phase == "formation" and not self.motion_started:
                    ready = all(v or w or (math.hypot(s["x"]-g[0], s["y"]-g[1]) <= .04
                                           and abs(math.remainder(s["heading"]-g[2], math.tau)) <= .035)
                                for (v,w), s, g in zip(commands, states, self.pose_controller.goals))
                    if ready:
                        self.motion_started = True
                    else:
                        commands = ZERO
            except (ValueError, ArithmeticError, RuntimeError) as error:
                self.get_logger().error(str(error))
                self.fault = "inference_error"
        if self.fault:
            commands = ZERO
        for index, publisher in self.commands.items():
            v, w = commands[index]
            message = Twist()
            message.linear.x, message.angular.z = float(v), float(w)
            publisher.publish(message)
        message = String()
        report = dict(phase=self.phase, fault=self.fault, odometry_fresh=fresh,
                      plan_id=self.plan_id, stamp_s=now)
        if self.robot_id is not None:
            report["command"] = [float(value) for value in commands[self.robot_id]]
        message.data = json.dumps(report)
        self.status_pub.publish(message)


def read_profile(path):
    profile = read(path)
    if type(profile["robot_id"]) is not int or profile["robot_id"] not in range(3):
        raise ValueError("robot_id must be 0, 1 or 2.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", profile["session"]):
        raise ValueError("session must begin with a letter and contain letters, digits or underscores.")
    if (not profile["odom_topic"].startswith("/") or profile["odom_topic"].startswith("/swarm/")
            or not all(profile[key] for key in ("odom_frame", "body_frame"))):
        raise ValueError("Configure the absolute driver topic and its odometry and body frames.")
    pose = profile["initial_pose"]
    if len(pose) != 3 or not all(type(v) in (int, float) and math.isfinite(v) for v in pose):
        raise ValueError("initial_pose must contain x (m), y (m), yaw (degrees).")
    return profile


def main():
    import argparse
    from rclpy.utilities import remove_ros_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path)
    args = parser.parse_args(remove_ros_args()[1:])
    profile = read_profile(args.robot_config) if args.robot_config else None
    rclpy.init()
    node = None
    try:
        node = MixedActorShadow(profile)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
