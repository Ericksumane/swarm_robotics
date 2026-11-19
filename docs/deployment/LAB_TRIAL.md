# First laboratory trial

Copy the same archive to each onboard computer.

1. With actuation disabled, record the ROS 2 version. Start the platform
   driver and run `ros2 topic list -t` to find its `nav_msgs/msg/Odometry`
   topic. Run `ros2 topic echo --once ODOM_TOPIC` using the actual topic
   name and record `header.frame_id` and `child_frame_id`. Record the driver,
   its motion command interface and the independent stop procedure.
2. Mark A and B on the floor. A is the origin and A to B points along +X.
   Positive Y is left of that direction. Measure each stationary chassis
   centre in metres and its heading in degrees, positive counterclockwise.
3. Follow the [configuration guide](MIXED_DISTRIBUTED_SHADOW.md). Use the
   same ROS domain and a new shared session name. Synchronize the clocks.
   Copy the example for each chassis ID to robot.json and enter the
   measured pose and actual driver topic and frame names.
4. Start one node per computer. Use `ros2 topic list` to check discovery and
   `ros2 topic echo /swarm/SESSION/shadow/robot_ID/status` to inspect each
   node. Replace SESSION and ID with the configured values. Compare the
   published poses to the floor measurements and check for
   `odometry_fresh: true` and `fault: null` in the status JSON. With three
   independent participants running, disconnect the observation laptop
   and use the onboard computers to confirm peer exchange continues.
5. With actuation disabled, interrupt one odometry stream after planning
   starts. Verify diagnostic commands go to zero and remain inhibited
   when the stream resumes. Restart all nodes with a new session.

With only the Go2 and WheelTec, check installation, calibration and peer
discovery. Team command calculation requires all three odometry streams.
A simulated Jackal can supply ID 2 if its clock and mission reference are
compatible. Its host must remain connected during that trial.

The shadow package publishes diagnostic commands. The movement release
requires agreement on the formation plan and mission phases, platform
motion adapters and verified stopping on hardware.
