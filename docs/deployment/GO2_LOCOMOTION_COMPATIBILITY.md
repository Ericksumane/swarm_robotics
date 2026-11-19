# Go2 locomotion compatibility

The JKK documents its platform as Go2 EDU/NX. The Unitree URDF and the
Gazebo gait URDF both expose twelve locomotion joints. The names differ.
The candidate name mapping is reported by
`scripts/check_go2_locomotion_contract.py`. Check axis signs, neutral
positions, limits and channel order before commanding joints.

The public Unitree Mjlab Go2 deployment configuration uses a 20 ms control
period, a twelve-joint mapping and a forward command range of -0.5 to 1.0 m/s.
Its observations include body angular velocity, projected gravity, velocity
commands, gait phase, relative joint positions and velocities, and the last
action. The matching export and deployment parameters are required for an
inference check. The current mixed Gazebo launcher uses a cruise target of 0.55 m/s
(0.32 m/s in severe). Validate the installed interface and physical
limits before selecting a bench-test speed.

Ask the lab for the installed locomotion interface and firmware or SDK version,
supported command ranges, and an anonymized record of commanded velocity,
odometry and joint states from a controlled straight run, turn and stop.
If the lab has a locomotion export, obtain it with its deployment configuration.

Sources

- https://github.com/jkk-research/JKK_UNITREE_GO2
- https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml
