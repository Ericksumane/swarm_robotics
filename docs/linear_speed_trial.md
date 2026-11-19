# Experimental linear-speed increase

At the time of this historical trial, defaults were 0.30 m/s forward and
0.15 m/s reverse. The subsequent requested standard and training campaign are
documented in speed033_training.md. The optional
--linear-speed-scale 1.1 selects 0.33 and 0.165 m/s in live simulation. Rotation
(0.8 rad/s), linear acceleration (0.3 m/s^2), angular acceleration (0.8 rad/s^2),
physics step, policy weights and mission deadlines remain unchanged.

The task capability inputs, action-to-velocity conversion, peer-history speed
bound and Gazebo DiffDrive forward/reverse limits use the same experimental
profile. Invalid multipliers outside [1.0, 1.15] are rejected. No command is
multiplied behind an unchanged actor capability or physics limit.

Deadline is 300 s in nominal/good/moderate and 420 s severe, counted from the
start of the episode including formation and alignment. Successful live hold
may continue beyond initial completion under the existing observation settings.

## Paired development trials

| Profile | Seed | Outcome with 1.1 | First completion (s) | Minimum sampled circular clearance (m) |
| --- | --- | --- | --- | --- |
| good | 2087500577 | success | 110.10000000000001 | 0.1116 |
| severe | 3624788268 | time_limit | None | 0.4326 |

Baseline user traces for these same seeds: good completed at 120.7 s with
minimum sampled clearance 0.1036 m; severe stopped at 420 s without starting
MOVE, minimum clearance 0.4757 m. These comparisons are development probes,
not final acceptance. New runs used live headless Gazebo physics with --fast;
reported mission times are simulated seconds, not accelerated wall time.
The trials stopped on first success and do not validate prolonged terminal hold.

The good case became faster, but increasing maximum speed cannot guarantee
faster formation or mission success. In the failed baseline severe case the
mean absolute formation speed was only 0.0233 m/s, well below the 0.30 m/s cap.
Missing information, slow learned acquisition and protocol waiting also matter.
This optional profile changes conditions from the trained nominal dynamics;
it is not promoted into the frozen delivery artifact or hardware runtime.

Example: bash scripts/run_delivery_gazebo.sh good --linear-speed-scale 1.1
Use an explicit --seed to compare against the default scale of 1.0. Omitting
--seed retains new random positions. Full traces are in
experiments/runs/linear_speed_trial_001. Synthetic circular clearance is not a
physical robot stopping guarantee.
