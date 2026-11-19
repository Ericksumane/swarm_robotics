# Nominal pose curriculum

Stage one uses near-equilateral starts with 2 cm position jitter and headings
sampled within 0.35 rad of world +X. Continue the fresh formation actor for
12,288 team steps from the corrected pose-contract pilot. The plan records the
source bundle digest. The other actors retain their weights.

Compare the candidate and its source on eight reserved seeds, 111101 through
111108, using the stage-one reset distribution and the same 60 s deadline,
nominal link profile, Go2 deadband and WheelTec steering response of 0.15 s.
Record full formation success, final geometry and heading separately, and
collisions. Promote only when the candidate reaches at least six successes,
matches or improves the source count, and has zero collisions.

The next stage increases position jitter and heading spread. Evaluate on
separate reserved seeds before attempting fully dispersed starts.
