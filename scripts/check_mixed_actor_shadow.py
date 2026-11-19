"""Exercise ROS odometry and shadow commands in an isolated ROS domain."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from mixed_actor_shadow import MixedActorShadow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    distributed = parser.parse_args().distributed
    rclpy.init()
    node = None if distributed else MixedActorShadow()
    processes = []
    if distributed:
        for i in range(3):
            processes.append(subprocess.Popen([
                sys.executable, str(Path(__file__).with_name("mixed_actor_shadow.py")),
                "--robot-config", f"configs/deployment/robot_{i}.example.json",
            ]))
    prefix = "/swarm/trial_001/shadow" if distributed else "/swarm/shadow"
    driver = rclpy.create_node("shadow_test_driver")
    executor = SingleThreadedExecutor()
    if node is not None:
        executor.add_node(node)
    executor.add_node(driver)
    status, commands = {}, {}

    pubs = []
    for i in range(3):
        driver.create_subscription(
            String, f"{prefix}/robot_{i}/status" if distributed else f"{prefix}/status",
            lambda m, i=i: status.update({i: json.loads(m.data)}), 10)
        suffix = "raw_odom" if distributed else "odom"
        pubs.append(driver.create_publisher(Odometry, f"/robot_{i}/{suffix}", 10))
        driver.create_subscription(
            Twist, f"{prefix}/robot_{i}/cmd_vel",
            lambda m, i=i: commands.update({i: (m.linear.x, m.angular.z)}), 10
        )

    def pump(seconds, send=True):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if send:
                for i, (x, y, yaw) in enumerate(((0, -.7, .2), (1.33, .04, .15), (0, .7, -.2))):
                    msg = Odometry()
                    msg.header.stamp = driver.get_clock().now().to_msg()
                    if distributed:
                        x, y, yaw = 4., -3., .7
                    msg.header.frame_id = "odom" if distributed else "mission"
                    msg.child_frame_id = "base_link" if distributed else f"robot_{i}/base_link"
                    msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
                    msg.pose.pose.orientation.w = math.cos(yaw / 2)
                    msg.pose.pose.orientation.z = math.sin(yaw / 2)
                    pubs[i].publish(msg)
            for _ in range(18):
                executor.spin_once(timeout_sec=.003)

    try:
        pump(4. if distributed else 1.5)
        assert len(status) == 3 and all(
            report["odometry_fresh"] and report["fault"] is None
            for report in status.values()), status
        assert len(commands) == 3 and any(v or w for v, w in commands.values()), commands
        for i in range(3) if distributed else (None,):
            name = f"mixed_actor_shadow_{i}" if distributed else node.get_name()
            topics = {topic for topic, _ in driver.get_publisher_names_and_types_by_node(name, "/")}
            ids = (i,) if distributed else range(3)
            assert {t for t in topics if t.endswith("/cmd_vel")} == {
                f"{prefix}/robot_{j}/cmd_vel" for j in ids
            }, topics
        print("PASS: fresh odometry and nonzero shadow-only commands", commands)
        pump(.8, send=False)
        assert all(r["fault"] == "odometry_stale" for r in status.values()) and all(
            (v, w) == (0., 0.) for v, w in commands.values()
        ), (status, commands)
        pump(.8)
        assert all(r["odometry_fresh"] for r in status.values()) and all(r["fault"] == "odometry_stale" for r in status.values()) and not any(
            v or w for v, w in commands.values()
        ), (status, commands)
        print("PASS: missing odometry zeros all commands; restored data does not rearm")
    finally:
        executor.shutdown()
        for process in processes:
            process.send_signal(signal.SIGINT)
        for process in processes:
            assert process.wait(timeout=5) == 0
        if node is not None:
            node.destroy_node()
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
