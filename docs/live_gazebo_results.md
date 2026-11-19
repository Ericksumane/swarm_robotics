# Live Gazebo integration verification

The runtime is an online feedback loop with Gazebo physics, not a recorded
trajectory viewer. Existing MAPPO weights were retained; no Gazebo retraining
or hardware-model validation is claimed.

## Engineering checks

- A physical actuator-only probe moved robot 0 by approximately 0.18 m and left
  the other two stationary. This scripted probe is not a learned-policy result.
- The paced three-second probe consumed approximately 3.07 wall seconds.
- Twenty simulation seconds of live policy feedback reduced the maximum pair
  error from approximately 1.08 m to 0.264 m.
- A full good-communication run with seed 42 reached final mission criteria at
  127.4 simulation seconds, using 1274 newly measured feedback/control steps.
  This verification used `--fast`; 28.6 wall seconds is not a 1x pacing claim.
- Goal errors audited from the physical MOVE-start positions were approximately
  [0.1253, 0.1434, 0.0977] m, consistent with the mission's reported errors.
- Maximum post-MOVE pair error was 0.11184 m, exceeding the 0.1 m tolerance.
  Therefore this is NOT a strict continuous-formation success. Minimum measured
  conservative footprint clearance remained approximately 0.2276 m.
- A continuous-hold check continued physics and actor inference through 150
  simulation seconds (1500 feedback steps), after first completion at 127.4 s.
  It retained fixed goals and active neural corrections rather than freezing
  the scene. Final goal errors remained below 0.2 m.
- All 56 project checks passed. Tests require fresh external feedback and prove
  that the Python unicycle integrator is not invoked by the live adapter.

These are engineering tests of a single learned-policy seed/profile, not a
success-rate estimate. Earlier Python-only rates must not be presented as
Gazebo rates. The graphical run is separately paced near the wall clock and
reports its actual real-time factor in the terminal log.

## Run and stop

```bash
cd ~/swarm_robotics
bash scripts/run_live_gazebo.sh --seed 42 --communication good
```

The world name is `swarm_live`. It contains three physical differential-drive
proxies with wheels and collision geometry. New velocity commands are computed
from live world-frame odometry. No recorded pose file is read and no pose service
is used to move robots. After success the GUI continues live HOLD. Close the
window to stop, or run:

```bash
cd ~/swarm_robotics
/usr/bin/python3 scripts/stop_live_gazebo.py
```

Use `--end-on-success` to stop the episode after arrival. `--time-limit 120`
sets the shorter original mission deadline; default startup deadline is 240 s.
See live_gazebo.md for architecture, source references and clock/odometry details.

The proxies are not verified models of the three laboratory robots. Odometry
uses ideal Gazebo world coordinates, and communication impairments remain the
configured application-level synthetic model. ROS 2 per-robot deployment,
localization realism and statistical physics-transfer evaluation remain future
work. The test harness centrally hosts independent local actors; it introduces
no robot leader and does not expose other robots' undelivered states to actors.
