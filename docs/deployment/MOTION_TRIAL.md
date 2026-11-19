# Supervised movement trial

This experimental package adds a ROS 2 Twist output gate to the onboard
formation, translation and final hold controller. The mission target is
5 m forward from the formed triangle, along the calibrated mission +X axis.

## Configuration and sequence

1. Read LAB_TRIAL.md and complete the shadow tests with actuation disabled.
   Use a fresh session for each run and synchronize all three clocks.
2. Confirm each actual driver accepts geometry_msgs/msg/Twist (linear.x
   in m/s and angular.z in rad/s). Check the real endpoint with
   ros2 topic info -v. Drivers that require TwistStamped, AckermannDrive
   or a vendor API need a compatible platform adapter. Never assume
   that a topic named cmd_vel is the right endpoint.
3. Verify an independent stop and a driver command watchdog on each chassis.
   Confirm the driver stops if its command publisher disappears. Use the
   vendor procedure to verify signs, limits, steering and braking on each
   robot individually. Clear the operating area and reserve an operator
   with access to the stop.
4. Add motion_topic, max_linear_mps and max_angular_radps to each robot.json.
   Use the verified driver topic and conservative limits within the
   executable caps of 0.20 m/s and 0.40 rad/s. The gate scales both axes
   together to preserve commanded curvature. Use one command source per driver.
5. Start the shadow controller on each robot, then start the gate in a
   second terminal on each. Use the commands below from the release folder.
   Stop the trial if behavior differs from the shadow expectations.

```bash
bash scripts/run_mixed_actor_shadow.sh --robot-config robot.json
bash scripts/run_mixed_motion_trial.sh robot.json --enable-supervised-motion
```

The gate outputs zero unless it sees three fresh, fault-free reports with
the same formation plan digest and phase. After motion begins it latches a
stop if freshness or plan agreement is lost. A phase disagreement inhibits
motion while the disagreement persists. Use the independent stop for an
unexpected motion. A process kill or power loss requires the driver watchdog.

## Validation limits

The phase gate compares reports from independent planners. It can block
indefinitely if their plans or phase transitions disagree. It does not
synchronize the translation reference tick by tick. Full physical acceptance
requires a successful trial with the three actual drivers and robots. This
bundle includes a generic Twist interface and has no vendor-specific
actuation implementation for incompatible drivers.
