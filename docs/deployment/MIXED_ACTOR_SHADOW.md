# Central mixed fleet shadow diagnostic

This diagnostic runs all four frozen actor components on one ROS 2 Jazzy
computer. It reads /robot_ID/odom for IDs 0 through 2 and publishes diagnostic
velocities under /swarm/shadow/robot_ID/cmd_vel. Use the
[distributed deployment guide](MIXED_DISTRIBUTED_SHADOW.md) for the onboard trial.

From the repository root, launch with:

```bash
bash scripts/run_mixed_actor_shadow.sh
```

The launcher uses ROS system Python with NumPy, rclpy, geometry_msgs,
nav_msgs and std_msgs. Publish poses in the common mission frame and twists
in robot_ID/base_link. Calibrate the poses before publishing them. The node
rejects stale data and latches a fault if valid odometry is lost after
planning starts. Inspect /swarm/shadow/status for freshness and faults.
