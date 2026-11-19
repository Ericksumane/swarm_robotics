# Platform verification before motion

| Platform | Information from the lab | Checks still needed |
| --- | --- | --- |
| Unitree Go2 Edu | Factory sensors and Intel D435i reported | Installed OS/ROS, odometry source and time basis, actual locomotion interface, limits and independent stop |
| WheelTec RosBot Pro | Ubuntu 24.04 and ROS 2 Jazzy reported | Exact steering geometry, body dimensions, steering/speed limits, odometry and independent stop |
| Clearpath Jackal J100 | Physical platform unavailable in this lab | Hardware access, exact equipment, installed software, odometry and stopping |

The mixed-fleet package first runs centrally on the Jazzy computer in
[shadow mode](MIXED_ACTOR_SHADOW.md). The three input streams need a shared
metric `mission` frame with +X in the mission direction and synchronized
timestamps. The software stops issuing nonzero diagnostic outputs when
odometry expires. This check uses no motor interfaces.

Record stationary localization and clock agreement first. Before supervised
movement, verify the supported vendor command and status interfaces, measured
body clearance, actual steering geometry, acceleration and deceleration,
watchdog behaviour and independent physical stopping. Use measured results
to decide whether the simulator geometry and motion settings require changes.
Validate each robot independently at the lab's agreed speed before a mixed
formation trial. Changing the hardware requires its own test evidence.
