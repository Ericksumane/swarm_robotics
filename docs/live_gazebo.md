# Live closed-loop Gazebo simulation

For the mixed fleet, use `bash scripts/run_mixed_realtime.sh severe` or choose
`nominal`, `good` or `moderate`. An optional numeric seed reproduces the
starting poses. The launch uses Gazebo physics and the selected frozen actors
with formation, alignment, 5 m translation and terminal recovery. See
[quick start](../GETTING_STARTED.md) for log locations and dependencies.

The paced controller targets RTF 0.98 with its wall-clock scheduler. It steps
the physics before reading the next odometry frame. Computation, ROS transport
or graphics that exceed the wall-clock budget reduce the measured RTF.
Use the logs to compare the measured value, especially under WSL.

## Historical generic-fleet harness

Run `bash scripts/run_live_gazebo.sh --seed 42 --communication good`.
This uses current Gazebo physics and newly computed MAPPO actions. It does not
read recorded paths and does not call a pose service to animate robots.
Initial poses are sampled once, used to construct the physical world, and then
allowed to settle under gravity. There are no subsequent programmatic teleports.

Data path:
Gazebo wheel/contact physics -> world-frame OdometryPublisher -> external-state
adapter -> local message impairment/receiver caches -> frozen actor selection
and MAPPO inference -> normalized-action velocity scaling -> DiffDrive cmd_vel.
The Python unicycle integration function is not called by this runtime.

The scene contains three generic differential-drive models with wheel joints,
mass/inertia, floor and robot collision geometry. They are synthetic proxies,
not verified Unitree Go2, Jackal or Wheeltec models. Odometry uses ideal world
poses from Gazebo, not wheel dead reckoning or a real localization system.

Control runs at 10 Hz simulation time with 200 Hz Gazebo physics. Gazebo is
stepped between control updates to guarantee fresh feedback and paused on
controller failure. The default runtime targets wall-clock speed, prints its
measured real-time factor, and never silently skips control intervals. GUI
play/step widgets are omitted because the controller owns clock advancement;
camera and entity inspection remain available. Pauses between short physics
chunks are normal. `--fast` disables pacing for headless verification only.

The default mission limit is explicitly 240 simulation seconds, matching the
separately labelled startup timing diagnostic. Use `--time-limit 120` to retain
the shorter budget. `--duration N` ends an engineering smoke test after N
simulation seconds; it does not count as task success. `--headless` omits GUI.
In GUI mode, a completed mission enters live HOLD: physics, odometry and actor
inference remain active with the same goals. The original deadline is checked
until first success; afterward interactive hold continues until the GUI closes
or Ctrl+C. This does not extend the time allowed to claim first mission success.
Collision/failure still stops commands. `--end-on-success` ends the episode instead;
`--headless --keep-controlling --duration N` verifies interactive hold without GUI.
After stopped episodes the GUI stays open until closed or Ctrl+C. Closing the GUI terminates only this run's owned children.
Every run has a unique GZ_PARTITION to avoid other Gazebo worlds.

The system-Python Gazebo driver and Python-3.10 inference worker communicate
over JSON lines because their native bindings use different Python versions.
A tiny separate service process isolates synchronous Gazebo clock requests
from odometry subscriber callbacks, avoiding observed request stalls. A missing
worker response, stale odometry or mismatched physics clock fails the run.

The simulation harness hosts independent local actors centrally for evaluation;
there is no robot leader. Actor inputs contain own state/mission information
and only neighbor messages delivered through the configured synthetic channel.
This is not yet distributed hardware deployment or actual Wi-Fi/DDS impairment.

Artifacts are recorded under experiments/runs/gazebo_live_*; latest run metadata
is experiments/runs/gazebo_live/latest.json. New control.jsonl files are outputs,
not trajectory inputs. Training weights and previous replay commands remain
unchanged. Earlier pure-Python success rates do not establish performance under
Gazebo physics; this transfer must be evaluated separately.

Official parameter references:
- https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1DiffDrive.html
- https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1OdometryPublisher.html

Physics-only actuator probe (scripted diagnostic, not a learned-policy result):
`bash scripts/run_live_gazebo.sh --headless --fast --probe --duration 3`.


## Explicit common delivery candidate

Use `bash scripts/run_delivery_gazebo.sh good --seed 42` from the project root.
Replace good with nominal, moderate or severe. This wrapper verifies the delivery
artifact checksum and selects the explicit profile deadline (300 s, or 420 s for
severe), with extended terminal observation. It performs live Gazebo wheel
physics, not video replay. The legacy run_live_gazebo defaults remain historical;
the delivery wrapper explicitly selects the current checksummed bundle.
The runtime-only installation archive excludes this development simulator.
