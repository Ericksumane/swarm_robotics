# Mixed fleet distributed shadow trial

See the [laboratory trial checklist](LAB_TRIAL.md) for the first session.

Run the same package on the Go2 (ID 0), WheelTec (ID 1), and Jackal (ID 2).
Each instance reads its own driver odometry, publishes the transformed pose
to its peers and publishes only its own diagnostic velocity. DDS carries the
messages directly on the real network. The portable computer is optional.
This trial requires three odometry sources, so a simulated Jackal may supply
ID 2 while the physical Jackal is unavailable.

## Install and configure

On a development system with ROS 2 Jazzy, install the system Python packages
for rclpy, NumPy, geometry_msgs, nav_msgs and std_msgs. Check the installed ROS
distribution and platform driver on each onboard computer before copying the
package. For a non-Jazzy install, validate package dependencies and DDS
interoperability before testing across machines.

Extract the same archive on each robot. Copy the corresponding
configs/deployment/robot_ID.example.json to a local robot.json and edit it:

- Keep the chassis ID. Use a new shared session name for every trial.
- Set odom_topic, odom_frame and body_frame to the driver's actual ROS names.
  The example topic and frame names are placeholders.
- Use floor marks A and B for the mission origin and +X direction. While
  stationary, fill initial_pose with the measured centre x and y in metres
  and chassis yaw in degrees. Y is positive left of A to B. Start while the
  robots are still at the measured poses. The first valid local odometry
  sample is mapped to this measured pose. Repeat calibration after a reset.
- Configure a common ROS_DOMAIN_ID and synchronize computer clocks. Verify
  peer topic discovery on the laboratory network. Keep the ROS graph isolated
  from unrelated publishers using the same session or IDs.

Start one node per robot from the extracted directory:

```bash
bash scripts/run_mixed_actor_shadow.sh --robot-config robot.json
```

Inspect /swarm/SESSION/shadow/robot_ID/odom and /status under the same robot
prefix. The diagnostic cmd_vel is under that prefix too. Never remap this
output to a vendor motor input. The 5 m goal is relative to the centroid at
formation completion, along mission +X. There are no simulated network
impairments in this launch. Check that the reported calibrated poses agree
with the floor measurements and that all three nodes see fresh data. Remove
one odometry stream and confirm the diagnostic outputs stop after freshness
expires once planning has started. Restore the stream and confirm the fault
stays latched. Restart all participants with a new session after a fault.

## Validation status

The host test launches three separate processes on one ROS 2 Jazzy host with
synthetic raw odometry. It checks calibration through nonzero command
production, one command topic per node, stale data handling and the fault
latch. Cross-computer discovery, clock synchronization and a moving mission
with synchronized plans and phase transitions remain unverified. Each
instance currently plans from its independently received samples. Different
snapshots can produce different plans or phase transitions. Before enabling
actuation, implement and verify peer agreement on the mission snapshot,
formation plan and phase transitions, then implement and bench-test the
platform motor adapters, watchdogs and independent stop mechanisms.
