# Dispersed formation precision entry

Individually settled robots can now enter precision correction when pair spacing
is outside tolerance. Translation still requires the precision geometry gate.

Severe Gazebo tests used reference 160k weights and seeds 3799001–3799003.
The previously blocked seed 3799001 formed and translated, then failed the
terminal alignment before the 135 s deadline. Full per-seed outcomes are in
experiments/runs/dispersed_precision_fix_001/results.json.

These complete dispersed starts differ from the near-formation planar tests.
The remaining failure needs terminal Ackermann pose recovery diagnosis.
