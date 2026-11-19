# Movement deployment candidate

This candidate prepares the installations for the movement release. Its
executable runs in shadow mode. Movement adapters and agreement on the
formation plan and mission phases are pending. Keep diagnostic commands
separate from platform motor inputs.

## Install on each robot

Use the same archive on all three computers. Extract it into a fresh
directory, verify its contents, and copy the configuration for the chassis.

```bash
tar -xzf swarm_mixed_deployment_candidate.tar.gz
cd swarm_mixed_deployment_candidate
sha256sum -c SHA256SUMS
cp configs/deployment/robot_0.example.json robot.json
```

Use robot_0 for the Go2, robot_1 for the WheelTec, and robot_2 for the Jackal.
The example above is for the Go2. Follow LAB_TRIAL.md for the initial shadow
trial and MIXED_DISTRIBUTED_SHADOW.md for ROS dependencies and networking.

## What to edit

| Setting | Where | When to change it |
| --- | --- | --- |
| Robot identity | robot.json, robot_id | At installation, to match the chassis. |
| Session name | robot.json, session | Before each trial. Use the same new name on all robots. |
| Driver feed | robot.json, odom_topic, odom_frame, body_frame | Match the platform driver's actual topic and frames. |
| Starting pose | robot.json, initial_pose | Measure while stationary before each trial. Enter x and y in metres and yaw in degrees. |
| ROS discovery | ROS_DOMAIN_ID in the launch terminal | Use the same laboratory domain on all robots. |
| Mission origin and heading | Floor marks A and B | A is the origin. A to B defines +X. Re-measure all starting poses when the marks move. |
| Travel distance | Current controller, fixed at 5 m | The target is 5 m beyond the formed triangle's centroid along +X. Distance and fixed endpoints are not configuration options in this candidate. |

The policy weights are selected by the bundle manifest and verified at
startup. The deployment uses live ROS traffic.

```bash
bash scripts/run_mixed_actor_shadow.sh --robot-config robot.json
```

## Collect platform information

With actuation disabled and the platform driver running, run the inventory
script on each robot. Save the output outside the release directory. Record
the exact platform model, firmware, vendor command interface, command
watchdog and independent stop procedure alongside it. The script also
accepts the actual driver odometry topic to report its type and publishers.

```bash
bash scripts/collect_deployment_inventory.sh /actual/odom > ~/swarm-inventory.txt
```

## Movement release criteria

1. Verify shared pose calibration, peer discovery, and operation without
   the observation laptop.
2. Verify the three onboard instances agree on the initial snapshot,
   formation plan and phase transitions under real network conditions.
3. Implement and bench-test the adapter for each platform using its
   reported driver interface. Verify command units and limits, local
   watchdog behavior and the independent stop.
4. Test each robot individually at low speed in a controlled area.
5. Validate formation, 5 m travel and aligned stop with all three robots.

The Jackal may be represented by a simulated odometry source during the
shadow trial. Full physical acceptance requires access to all three platforms.
