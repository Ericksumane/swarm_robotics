# WheelTec terminal recovery budget

The severe dispersed case 3799001 originally reached a 135 s driver duration
limit during terminal Ackermann recovery. The final pose was not settled.
The terminal phase now receives up to 90 simulated seconds within the
configured total mission deadline, allowing its forward/reverse recovery arcs.

Three subsequent Gazebo repetitions of the same initialization completed at
100.2, 118.1 and 141.7 simulated seconds with the 160k reference actors.
Maximum final pair errors were 4.46, 1.36 and 4.73 cm respectively.
Maximum heading errors were 3.37, 2.92 and 1.45 degrees respectively.
All three reported settled and approximately 5 m translation.

The first two completions occurred before the original limit. The third
required more than 135 s. These observations support additional terminal
time for this case but do not establish a general recovery success rate.
Same-seed physics/ROS runs varied. Raw results are recorded in
experiments/runs/wheeltec_terminal_budget_001/results.json.
