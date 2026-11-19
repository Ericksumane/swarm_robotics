# Mixed-fleet delivery and laboratory preparation

Use the 160k manifest as the frozen reference. Keep the four weights, actuator
settings, simulator model versions, communication profiles and seed list
together when reporting results. Keep the generic MAPPO runtime separate.

The one-million-step fine-tuning campaign enables terminal pose recovery in
the planar environment. Compare its weights against the reference on reserved
seeds with recovery enabled for both. Record success, timeouts and clearance
failures by profile and phase. Before promoting weights, run complete paired
Gazebo missions from dispersed positions. Record time, minimum separation,
5 m travel, final pair errors, heading, and rest. Retain the reference if the
candidate regresses or the results are inconclusive.

Use the [quick start](../GETTING_STARTED.md) to run the four simulator profiles.
The pacing target is 0.98. The terminal and run summaries record measured RTF.
Keep the same simulated control and physics intervals when comparing policies.

Use the [mixed-fleet shadow package](deployment/MIXED_ACTOR_SHADOW.md) for
the first lab session. Record the actual Go2 and WheelTec odometry topic,
message type, frame, timestamp and rate with the robots stationary. Confirm
that both positions and headings agree with marked poses in a common frame.
A Jackal can supply simulated odometry for this acceptance step only when
its mission coordinates and time basis are aligned with the real streams.

Confirm each robot's vendor command interface, stopping and limits before
implementing actuator adapters. The first supervised movement is a separate
stage requiring the lab's independent physical stop and measured limits.
See [platform verification](deployment/HARDWARE_PENDING.md).
