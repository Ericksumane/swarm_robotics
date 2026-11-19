# External reference review and geometry decision

## Checked primary repositories

| Repository | Useful contribution | Compatibility boundary |
|---|---|---|
| https://github.com/marlbenchmark/on-policy | Official MAPPO optimizer and reference training choices | Already pinned; project collector and actor remain extensions |
| https://github.com/proroklab/VectorizedMultiAgentSimulator | Vectorized 2D multi-robot scenarios, custom tasks and rewards | Simulator reference, not validated real-robot transfer |
| https://github.com/facebookresearch/BenchMARL | MAPPO benchmarking, standard evaluation and checkpointing; VMAS examples | TorchRL implementation; use as independent reference, not silently replace the required upstream implementation |
| https://github.com/OPT4SMART/ChoiRbot | ROS 2 heterogeneous peer-to-peer coordination without central coordinator | README requires ROS 2 Dashing; Jazzy compatibility must be checked |
| https://github.com/robotarium/robotarium_python_simulator | Unicycle examples, transformations, geometry and collision tools | Classical control/testbed reference, not a MAPPO solution |
| https://github.com/LorenzoBalandi/Robot-Formation-Control | Bearing-based 2D formation examples in ROS 2 | Course project using ROS 2 Foxy and classical laws, not MAPPO |

These repositories were inspected as references. No external source code or new
dependencies are imported into the training environment by this review.
The VMAS README specifically documents navigation, flocking and passage tasks;
it is not being represented as an out-of-the-box solution to this exact mission.

## Triangle versus line

The implemented three-agent task asks for all three pair distances to equal 1m.
That is an equilateral triangle. A line with consecutive spacings of 1m requires
distances d01=1m, d12=1m, d02=2m (up to the chosen agent order). Thus a line
cannot be obtained by changing the formation label while retaining the current
all-pairs equal-distance reward, actor features and success metric.

Use explicit templates or a desired-distance matrix for a future configurable
geometry contract. Triangle template: [(-.5,-sqrt(3)/6),(.5,-sqrt(3)/6),(0,sqrt(3)/3)].
Line template: [(-1,0),(0,0),(1,0)]. Robot ordering/slot assignment is a formation
role, not a designated motion leader. Fixed world targets require a common frame
and an explicit mission reference available to every actor. Changing geometry
requires separate training and evaluation; line formation is not assumed easier.

## Localization boundary

The simulator currently supplies ideal local pose estimates in a shared frame.
Its geometric failures therefore do not establish a sensor-positioning problem.
Hardware odometry from different robots normally has separate origins: exchanging
poses alone is insufficient without frame alignment. A shared localization frame
must be established and measured before applying the trained policy to the fleet.

## Extended training protocol

Use a 120-second formation-only training limit, matching the diagnostic that
gave 20/40 nominal and 25/40 good successes for the frozen policy. Resume both
actor and critic plus optimizers from the selected source checkpoint; use fresh
episode/sample RNG seeds. This is statistical continuation, not an exact replay
of the interrupted training state. Keep the triangle, existing reward and robot
limits fixed. Train for 200000 additional environment steps and select among
checkpoints using disjoint development seeds; compare against the frozen policy
on new common final seeds. Preserve the original 120-second/30-second evidence.

Increasing the allowed episode duration changes the task budget; it is not a
policy improvement by itself. Do not claim this formation-only budget validates
the full five-meter translation mission. No unlimited-duration success criterion
or relaxed geometric tolerance is introduced.

## Concrete reference-derived verification

The VMAS DiffDrive implementation was inspected at:
https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/simulator/dynamics/diff_drive.py
It uses forward velocity and yaw-rate commands with the standard unicycle
equations. scripts/check_unicycle_reference.py independently integrates those
equations and checks our bounded-command simulator at constant speed, including
reversing and turning. This is an equation-level cross-check, not execution of
the external VMAS engine and not evidence that MAPPO learns the task.
No classical motion controller is introduced into the policy.
