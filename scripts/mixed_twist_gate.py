"""Gate onboard commands for a supervised ROS Twist driver trial."""
import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from mixed_actor_shadow import read_profile


class TwistGate(Node):
    max_age = .30

    def __init__(self, config):
        self.robot_id = config["robot_id"]
        super().__init__(f"mixed_twist_gate_{self.robot_id}")
        self.output = self.create_publisher(Twist, config["motion_topic"], 1)
        self.linear_limit = config["max_linear_mps"]
        self.angular_limit = config["max_angular_radps"]
        self.peers = {}
        self.active = False
        self.latched = False
        prefix = f"/swarm/{config['session']}/shadow"
        for index in range(3):
            self.create_subscription(
                String, f"{prefix}/robot_{index}/status",
                lambda msg, i=index: self.receive(i, msg), qos_profile_sensor_data)
        self.create_timer(.05, self.tick)

    def receive(self, index, message):
        try:
            data = json.loads(message.data)
            stamp = data["stamp_s"]
            command = data["command"]
            if (not math.isfinite(stamp) or len(command) != 2
                    or not all(math.isfinite(v) for v in command)):
                raise ValueError("Non-finite status")
            if index in self.peers and stamp <= self.peers[index][0]["stamp_s"]:
                return
            self.peers[index] = (data, time.monotonic())
            if data["fault"] is not None:
                self.latched = True
        except (TypeError, ValueError, KeyError):
            self.latched = True

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        monotonic = time.monotonic()
        fresh = (len(self.peers) == 3 and all(
            item["odometry_fresh"] is True
            and -.05 <= now-item["stamp_s"] <= self.max_age
            and monotonic-received <= self.max_age
            for item, received in self.peers.values()))
        reports = [item for item, _ in self.peers.values()]
        plans = {item.get("plan_id") for item in reports}
        agreed_plan = len(plans) == 1 and None not in plans
        if self.active and (not fresh or not agreed_plan):
            self.latched = True
        phases = {item.get("phase") for item in reports}
        output = Twist()
        if (fresh and agreed_plan and not self.latched and len(phases) == 1
                and phases <= {"formation", "translation", "terminal"}):
            v, w = self.peers[self.robot_id][0]["command"]
            scale = max(1., abs(v)/self.linear_limit, abs(w)/self.angular_limit)
            output.linear.x, output.angular.z = v/scale, w/scale
            self.active |= bool(v or w)
        self.output.publish(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--enable-supervised-motion", action="store_true")
    args = parser.parse_args()
    config = read_profile(args.config)
    topic = config.get("motion_topic")
    if (not args.enable_supervised_motion or not isinstance(topic, str)
            or not topic.startswith("/") or topic.startswith("/swarm/")
            or topic == config["odom_topic"]):
        parser.error("Provide an explicit driver Twist topic and --enable-supervised-motion.")
    for key, ceiling in (("max_linear_mps", .20), ("max_angular_radps", .40)):
        value = config.get(key)
        if type(value) not in (int, float) or not 0 < value <= ceiling:
            parser.error(f"{key} must be positive and at most {ceiling}.")
    rclpy.init()
    node = TwistGate(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.output.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
