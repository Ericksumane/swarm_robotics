# Implementation status

The current mixed-fleet Gazebo reference is the 160k checkpoint pinned in
`experiments/mixed_campaign_160k_candidate.json`. Two paired seeds completed
the mission per profile after the severe formation confirmation fix.
That sample is a limited acceptance check, with a simulated Jackal and
provisional WheelTec dynamics.

A diagnostic found that all 32 planar moderate/severe failures reached the
end of the motion reference without meeting the terminal spacing tolerance.
With the same weights and terminal pose recovery enabled, 14/16 moderate and
11/16 severe arrivals succeeded on those seeds. A 20k follow-up with recovery
enabled improved severe formation on a separate holdout from 4/16 to 7/16 and
arrival from 12/16 to 13/16, with one fewer good formation success.

The completed campaign collected one million additional planar environment
steps from the 160k reference, 250k per profile, with terminal pose recovery
enabled. The final checkpoint is experimental. The simulation and lab
package retain the 160k reference because the final weights regressed in
moderate formation. See [campaign results](mixed_arrival_1m_results.md).
Its manifest, progress and checkpoints are in
`experiments/runs/mixed_160k_arrival_campaign_1m_001`.
The packaged reference remains subject to passive odometry validation in
the lab. See [current deployment guide](deployment/MIXED_ACTOR_SHADOW.md)
and [platform work](deployment/HARDWARE_PENDING.md).

The original distributed ROS MAPPO runtime, typed peer transport and CPU
installer are documented separately in [deployment](deployment/README.md).

## Delivery checks

The repository-wide check runner completed 73 Python checks. The WheelTec
waypoint pilot, Go2 joint-name inspection and ROS 2 shadow odometry test ran
separately because they require artifacts or a ROS setup. The historical
WheelTec waypoint pilot completed its diagnostic and reported failures in its
experimental controllers. It is separate from the selected WheelTec arc actor. Static syntax
verification covered first-party Python and the top-level shell launchers.

A paced nominal Gazebo run completed at seed 196802 with a target RTF of 0.98
and measured RTF of 0.258 on the development WSL system. This was measured
before shortening the mixed bridge pause for the 0.98 mode. Run Gazebo
instances one at a time. Concurrent launches contended for the ROS 2
controller spawner and caused startup failures.

Final paced Gazebo checks completed all four profiles on seed 196802.
See [the results table](mixed_arrival_1m_results.md) for measured RTF and travel.

The 5m campaign completed and retained the 160k reference. The controlled
moderate regression diagnosis isolated the Jackal pose actor. See
[mixed campaign results](mixed_campaign_5m_results.md).
