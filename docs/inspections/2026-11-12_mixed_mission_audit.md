# Mixed-platform mission audit

Audit date 2026-11-12. Scope covers the current WSL workspace, the selected generic-kinematics baseline, the 600,000-step campaign, the mixed Gazebo runner, and the ROS shadow deployment. Training and policy selection were left untouched.

## Evidence and delivery status

The intended task starts three robots from dispersed planar poses, builds an equilateral triangle with 1.20 m center spacing, aligns body headings to the mission direction, moves 5 m, and stops with position, orientation and formation inside tolerance. The 1.60 m/s setting is a forward command limit. Achievable cruise speed on each platform remains a separate measurement. Full mixed-platform MAPPO mission acceptance remains open. The cooperative Gazebo reference provides a simulation demonstration with a programmed controller. The ROS runtime publishes shadow commands.

## Findings

1. **The campaign uses generic plant dynamics.** The saved tasks under `experiments/runs/mixed_speed160_spacing120_600k_001` omit `mixed_command_limits` and `rosbot_steering_time_constant_s` in every stage. The formation stages also omit `formation_pose_goal` and platform capabilities. `Simulation.advance` only applies the mixed constraints when enabled. The campaign scales forward speed and spacing while retaining the original generic stage tasks. The live runner applies a Go2 deadband and a WheelTec curvature bound, then advances physical Gazebo models. The WheelTec model remains a provisional Ackermann approximation. A generic actor can request turning in place during training, even when the WheelTec will need translation and room to turn in execution.

2. **The cruise training objective and deployed control flow differ.** The trainer samples one actor throughout each stage. Both cruise experts stay active even past the point where deployment would route to approach. The actor's observed goal vector saturates at 0.50 m and the deployed switch uses unclipped distance with a 0.70 m horizon. All 475 completed cruise and degraded-cruise training episodes ended outside the area. In a deterministic nominal replay of the campaign cruise reset, seed 416020, the final cruise actor alone reached 9.185 m centroid displacement and exited at 43.0 s. With the same reset and final actors plus normal routing and supervisors, adding the required history observation configuration, approach first engaged at 11.0 s. The team reached 5.123 m centroid displacement and timed out at 120 s. Final individual goal errors were 0.255, 0.146 and 0.168 m. Each robot had previously entered within 0.035 m of its goal at some point, which gives no guarantee of simultaneous arrival. This replay demonstrates two failures under different control flows. It does not attribute every outside-area event to one mechanism.

3. **Arrival needs a joint position and heading recovery contract.** MOVE enters HOLD within 0.15 m and returns to MOVE outside 0.20 m. The parking latch requires individual distance, speed, yaw rate and heading evidence. Its release uses individual position and heading. Team success separately requires every destination within 0.20 m, formation within 0.10 m, all headings within 5 degrees and continuous stationary hold. A locally parked team can violate global geometry, as the 48 severe training parking deadlocks demonstrate. In the aligned planar benchmark the new candidate stopped approximately 0.16 to 0.27 m beyond the longitudinal target, with a maximum position error of 0.279 m and heading error of 36.1 degrees. The original weights under the same 1.60 m/s task had 0.374 m maximum position error. The generic reference at 0.33 m/s reached 0.119 m and 20.6 degrees in this 1.20 m, 5-degree benchmark. None completed. The two nominal seeds produced identical trajectories because the benchmark forces the same initial poses. A fixed 0.70 m switch provides no explicit braking-distance guarantee. At the task's nominal deceleration of 0.30 m/s², the ideal stopping distance v²/(2a) is 1.07 m at 0.80 m/s and 4.27 m at 1.60 m/s, before feedback delay. Use these as design checks and measure actual stopping distances per platform.

4. **The live random-start failure happened in formation.** Run `gazebo_live_20261111_234559_23849`, seed 795485140, spent all 481 samples in FORM. It reduced maximum pair error from 1.623 to 0.338 m, with minimum sampled circular clearance 0.320 m. Final heading errors were 153.7, 128.5 and 34.6 degrees. No readiness certificate or movement goal existed at the end. The GUI closed at 48.1 simulated seconds after 197.9 wall seconds. This run gives no evidence of live cruise or braking performance. The separate locally labelled formation candidate reached 4/8 near-triangle planar successes without collisions. That result has a narrower start distribution and belongs to a different checkpoint. The previous raw-pose layout issue identified in the Claude inspection has already been addressed, and the current input-parity check passes.

5. **Severe communication exposes both observability and coordination limits.** The synthetic profile uses 10 Hz, 0.30 s base delay, 0.20 s jitter, burst losses, outages and a 500 B/s link budget. Even 64 B at 10 Hz offers 640 B/s per link before extra mission metadata. With 0.50 s maximum information age, queueing and serialization can consume freshness. Both formation stages scored zero severe successes across 64 completed training episodes. The isolated alignment actor recorded 53/62 severe successes, which does not validate distributed readiness from dispersed starts. Start proposals require peer evidence and clocks. The safe behavior under a partition needs to be explicit. Physical Wi-Fi characteristics and independent ROS processes require separate validation. The Gazebo Python worker batches observations in one process and models six directed links in software. ROS shadow nodes exist for exercising the decentralized interface.

6. **The Gazebo viewer and physical transfer have independent limitations.** The live run achieved a real-time factor of 0.243 against a target of 0.99. The mixed loop includes five mandatory 20 ms simulated advances and five wall sleeps per control tick, plus service and physics overhead. The target flag is a pacing cap and does not establish performance. Hardware profiles require command adapter verification, common-frame odometry, time synchronization, measured limits and braking, stale-data behavior, watchdog and independent stop. The lab confirmed Go2 EDU and WheelTec availability. The Jackal remains simulated in this lab context, per the information supplied by the user. A successful three-platform simulation cannot establish a physical three-robot demonstration.

## Priorities and finish criteria

1. Freeze the selected baseline and the 600k checkpoint as reproducible artifacts. The selected generic checkpoint is byte-identical across the two project folders. Preserve exact task, policy digest, simulator assets, launch command and runtime dependencies. The mixed workspace still contains substantial uncommitted work and local simulator assets. Use the programmed cooperative mission as the Gazebo reference and label its controller in the deliverable.
2. Pass the isolated nominal MOVE to HOLD benchmark first. Use the actual routing and stop logic during data collection and evaluation. Compare both checkpoints on identical starts while varying pose and orientation across trials. Measure the trigger distance, stopping error, heading error, clearance and sustained hold. Correct the braking envelope and recovery behavior before another large training campaign.
3. Pass nominal formation from dispersed starts with the mixed plant constraints enabled. Target position and heading jointly, especially for WheelTec. Validate training against measured or Gazebo-derived actuator response and record supervised intervention counts. Test the startup handshake separately from formation control. Merge phases only after each passes on held-out cases.
4. Test the complete mission with held-out starts in nominal, then good, moderate and severe. Record failure by phase, continuous terminal hold, and sampled clearance. If safe progress cannot be guaranteed during an outage, report controlled stop or hold separately from full mission success.
5. Release the simulation result with a single verified launch path and an English installation guide. Keep physical qualification open until the platform interfaces and lab trials produce evidence.

Verification during the audit included the saved training progress and six stage task snapshots, the nominal aligned comparison, the live control trace, a deterministic cruise comparison, and passing targeted checks for formation input parity, vector GAE boundaries, local parking behavior and external feedback discipline. No full test suite, new Gazebo campaign or physical trial was completed during this audit.

## Follow-up corrections

Two successful afternoon Gazebo runs used the programmed cooperative reference
and seed 8605. The learned local-pose candidate separately completed one
formation-only live run on seed 1845727269. The final 600k mixed live run used
a different checkpoint, a different seed and the full mission. These runs
cannot establish a paired regression.

Both acquisition routing and history startup now use the task's desired
spacing. Previously the 1.20 m task was compared against 1.00 m in both
decisions. The nominal stationary 1.20 m regression fixture reaches MOVE
and assigns a 5 m goal to each robot. Formation pose features, heading reward
and formation success now reference the mission direction in all paths.

The optional forward braking envelope is shared by the simulator, live worker
and ROS runtime. The cap is sqrt(2 a max(0, d_parallel - max(0, v) dt)).
The nominal aligned planar comparison with the 0.64 s provisional steering
response reduced maximum final position error from 0.293 m to 0.089 m and
maximum pair error from 0.147 m to 0.047 m. The three robots reached HOLD,
but heading still exceeded 5 degrees and the episode timed out.

The experimental combination of the earlier pose actor and the braking
envelope failed both dispersed planar mission runs. It improved heading on
both and worsened geometry on one. The numerical seeds are reused identifiers
from Gazebo; the evaluator samples planar starts and does not replay exact
Gazebo odometry. Results and the nominal 4096-step dispersed formation pilot
are stored in experiments/runs/mixed_mission_repair_001. No checkpoint was
promoted.
