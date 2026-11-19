# Nominal pose contract pilot

Train the fresh-message formation actor for 30,000 three-robot environment
steps with the existing nominal pose configuration. Compare the candidate
against its source bundle. Other actors retain their source weights.

- Plan `configs/training/mixed_formation_pose080.json`
- Source `experiments/runs/mixed_formation_capabilities080_001/candidate_bundle.pt`
- Source SHA-256 `3241703387c8a5731aa66ae2d0b985490219ed783ab0b20615195e94f9e6b1e0`
- Paired evaluation seeds 110601 through 110612
- Mixed planar dynamics with the Go2 deadband, WheelTec curvature limit,
  and 0.15 s WheelTec steering response for both bundles
- Dispersed starts with a minimum initial pair distance of 1.65 m
- A 60 s evaluation deadline with the same formation and heading criteria
  for both bundles

Advance to a Gazebo policy trial when the candidate completes formation
in at least 9 of 12 cases, matches or exceeds the source success count,
and records no collisions. Compare per-seed error, minimum circular
clearance and completion time even when the candidate misses this gate.
This pilot measures formation and alignment. A Gazebo trial separately
measures transfer to the mixed physical models.
