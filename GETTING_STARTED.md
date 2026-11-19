# Mixed-fleet quick start

Complete the [simulation setup](docs/deployment/SIMULATION_SETUP.md), then open
a terminal in the project directory.

```bash
cd ~/swarm_robotics
bash scripts/run_mixed_realtime.sh nominal
```

Use `good`, `moderate` or `severe` in place of `nominal`. For reproducible
poses run, for example, `bash scripts/run_mixed_realtime.sh severe 196802`.
Add `--headless` after the profile and optional seed to run without a window.
Stop the current run with Ctrl+C before launching another on the same ROS domain.

The launcher sources ROS 2 Jazzy and `go2_gait_ws/install/local_setup.bash`.
Use the prebuilt local Go2 gait overlay. If it is absent, use the build and
compatibility notes in [Go2 locomotion](docs/deployment/GO2_LOCOMOTION_COMPATIBILITY.md).
The launcher uses the ROS system Python and loads the JSON actors directly.
It can run from the same shell whether the training virtual environment is
active or inactive.

The terminal prints the output directory. After a run, check `manifest.json`
for the seed and settings, `control.jsonl` for the measured poses and commands,
and `summary.json` for completion reason, distance, final errors and RTF.
For GUI startup problems, check `gui.log` and the
[WSL diagnostic notes](docs/realtime_and_wsl_diagnostics.md).

## Compare the experimental weights in Gazebo

The default launcher runs the 160k reference. To inspect the experimental
final checkpoint in simulation, select its separate manifest explicitly.

```bash
SWARM_ACTOR_MANIFEST=experiments/mixed_arrival_1m_experimental.json \
  bash scripts/run_mixed_realtime.sh severe 196802
```

Use the same seed and profile without the environment variable to compare.
See the [campaign results](docs/mixed_arrival_1m_results.md) for the selection decision.

## Training

The training command below warms up from the frozen 160k checkpoint and
collects 250,000 planar steps per profile with terminal pose recovery active.
Choose a new output directory for each reproduction. Training uses a Python
environment with PyTorch and NumPy.

```bash
cd ~/swarm_robotics
export PYTHONPATH="$PWD/src:$PWD/scripts"
python -u scripts/train_mixed_actor_campaign.py \
  policies/mixed_campaign_160k experiments/runs/mixed_arrival_reproduction_1m \
  --steps-per-profile 250000 --recover-arrival --evaluation-interval 25
```

`progress.json` records completed steps and the latest checkpoint. A checkpoint
contains four JSON actors. Intermediate planar evaluations are development
checks. Compare candidates on reserved seeds and in full Gazebo missions
before replacing the pinned simulation manifest or rebuilding the lab package.

Run the lab test using [the mixed-fleet shadow guide](docs/deployment/LAB_TRIAL.md).
The older generic MAPPO commands and results are recorded in
[training environment](docs/training_environment.md) and
[common ROS deployment](docs/deployment/README.md).

## Final randomized mission checks

Use the four profile commands at the top of this guide for new random starts.
The terminal prints a replayable seed and the output directory. Let the run
continue until the JSON summary reports its completion reason. A normal
completed mission reports cooperative_mission_success. The GUI stays open
after stopping, allowing the final formation to be inspected.

The launcher uses frozen 160k actors, local Go2 feedback, initial anchor yaw
preparation when needed, precision geometry recovery, and up to 150 simulated
seconds for dispersed formation within a 240 s mission budget. The group
reference cruise speed is 0.55 m/s, capped at 0.32 m/s for severe.
Use the default physics batch size for the final demonstration. GPU graphics
selection and the 0.98 wall-clock pacing target are applied by the launcher.
Measured RTF depends on host load.

See [randomized acceptance](docs/final_random_acceptance.md) for the finite seed
sample and the failures found and repaired during that check.
