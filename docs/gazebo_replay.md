# Gazebo trajectory replay

This viewer uses three synthetic circular markers matching the initial
0.25 m footprint radius. They are not Go2, Jackal or RosBot physical models.
Gazebo UserCommands places static markers at recorded evaluation poses.
No MAPPO inference, dynamic contact response or ROS controller runs here.

Open two Ubuntu terminals, source /opt/ros/jazzy/setup.bash in both, then:

Terminal 1, from the repository root:
`gz sim -r gazebo/replay.sdf`

Terminal 2:
`python3 scripts/replay_gazebo.py experiments/runs/formation_proximity_001 --speed 2`

Use system python3 from the sourced ROS installation, not .venv, because
Gazebo transport bindings are supplied for the system interpreter.
Ctrl+C stops playback. It loops by default; --once plays one episode.
The viewer intentionally reproduces collisions from the failed experiment.
Actual Gazebo dynamics and platform-specific models remain separate work.

## Recommended single command

`bash scripts/view_gazebo.sh`

Starts an isolated server and GUI, waits for the world service, then plays
the recorded trajectory. Ctrl+C cleans up all child processes. Logs are
experiments/runs/gazebo_viewer/server.log and gui.log.

## Replay rendering configuration

The replay world uses a 0.01 s physics step instead of the implicit default,
a target real-time factor of 1.0, disabled world shadows, and a project-local
GUI configuration framing the recorded trajectories. This reduces unnecessary
server updates for static pose playback; it does not alter training dynamics
or demonstrate 98% performance. Actual RTF depends on host load and rendering.
