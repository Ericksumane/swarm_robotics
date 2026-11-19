# Development checks and troubleshooting

Build on the verified Ubuntu 24.04 / Jazzy workstation:

```bash
bash scripts/build_ros.sh
bash scripts/test_ros_shadow.sh
```

The second command is a development-only three-process ROS smoke test. It
publishes a stationary odometry fixture, checks scheduled starts and nonzero
shadow requests, then removes one odometry stream and checks all three stop.
It does not move physical robots or prove full-trajectory performance. The
script exists in the development repository, not the minimal robot package.

Common messages:

- `awaiting_peers`: no motion until both expected peer sessions are enrolled;
  check IDs, mission IDs, ROS domain/network discovery and topic names.
- Odometry frame mismatch: provide the actual common-frame transform through
  a verified localization adapter. Renaming a frame does not align coordinates.
- Stale/future odometry or peer time errors: check original timestamps and
  clock synchronization. Do not stamp old measurements as new to bypass checks.
- Peer session changed: a peer restarted. Stop/restart the whole mission with
  a new common ID; do not silently reuse stale certificates.
- Mission service timeout: inspect experiments/runtime logs, CPU timing and
  dependency versions. Do not disable the watchdog to conceal a missed deadline.
- Checksum mismatch: restore the matching artifact/manifest pair or install a
  separately verified release. Never mix half-updated release directories.

Record the manifest/policy hash, robot profile, initial poses, localization and
clock checks, real network measurements, commands, odometry and stop events for
each laboratory trial. Keep failed runs. Collect ROS bags and measured timings
before attributing a failure to reinforcement learning or radio conditions.
