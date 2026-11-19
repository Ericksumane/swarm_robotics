# Go2 local feedback in good communication

User seed 2112018943 timed out at 90 s during precision formation with
Go2 heading around -31 degrees. The coordinator supplied local Go2 feedback
only in severe, whereas the planar trainer always supplied it. The Gazebo
coordinator now supplies local Go2 odometry in each profile. Other robot
feedback selection remains profile dependent. The same seed completed
formation, translation and terminal recovery at 71.4 simulated seconds in
a headless test. Full records are in experiments/runs/good_go2_local_feedback_001.
This is a single-seed recovery test. Broader random-start acceptance is pending.
