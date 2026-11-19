# Randomized complete mission verification

Two random initialization seeds were generated for each of four profiles.
Failures found during the campaign were retained in their original directories.
The good start deadlock was corrected by allowing preparatory anchor yaw.
The moderate geometry mismatch now signals precision entry.
The dispersed formation budget is up to 150 s within a 240 s mission deadline.
The severe initialization failure was repeated after controller setup.
The repaired seeds were replayed; this table reports those latest attempts.
The other successful seeds were run before these latest coordinator changes.
This is a development/replay result, not an independent clean final-code benchmark.

| Profile | Seed | Reason | Simulated seconds | Minimum centre distance (m) |
| --- | ---: | --- | ---: | ---: |
| nominal | 227093656 | cooperative_mission_success | 80.5 | 1.009 |
| nominal | 1555297313 | cooperative_mission_success | 112.5 | 0.696 |
| good | 676531854 | cooperative_mission_success | 106.1 | 0.777 |
| good | 1251066829 | cooperative_mission_success | 79.8 | 0.906 |
| moderate | 2895921022 | cooperative_mission_success | 89.7 | 0.793 |
| moderate | 1073831986 | cooperative_mission_success | 93.6 | 1.022 |
| severe | 4073149796 | cooperative_mission_success | 112.7 | 0.891 |
| severe | 3403002117 | cooperative_mission_success | 176.0 | 0.711 |

The headless checks used fast clock stepping and the same 1 ms physics step
and 10 Hz policy interval. The interactive launcher targets RTF 0.98.
New random starts can still expose failures. Frozen trained actors operate
with programmed planning, coordination and precision recovery.
Physical deployment requires common-frame odometry and vendor adapter tests.
