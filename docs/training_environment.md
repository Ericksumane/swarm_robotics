# Training Environment

## Verified configuration

- Platform: Ubuntu 24.04 through WSL 2.
- Environment: .venv, managed with uv.
- Python: 3.10.22.
- NumPy: 1.26.4.
- PyTorch: 2.5.1+cu121.
- PyTorch CUDA runtime: 12.1.
- GPU: NVIDIA GeForce GTX 1650, 4 GB.
- MAPPO source: external/on-policy.

## Completed checks

- CUDA device detection: passed.
- GPU tensor computation: passed.
- GPU gradient backpropagation: passed.
- Official MAPPO trainer import: passed.
- Official MAPPO policy import: passed.

These checks do not yet validate policy construction, training updates,
environment integration, or deployment.

## Activate for training

```bash
cd ~/swarm_robotics
source .venv/bin/activate
unset PYTHONPATH
unset PYTHONHOME

```

Use a separate terminal for ROS 2.

## Dependencies

The installed package snapshot is stored in:
configs/training/requirements-installed.txt

The upstream legacy requirements.txt and environment.yaml were not
installed wholesale. Compatibility with this environment is being
validated incrementally.

## Policy integration check

Command: python scripts/check_mappo_policy.py

- Device: CUDA.
- Shared actor evaluated on three synthetic agent observations.
- Action shape: (3, 2).
- Critic value shape: (3, 1).
- Finite outputs: passed.
- Local-only actor inference: passed.

This check uses an untrained feedforward policy.
It does not validate learning, recurrent memory, variable team sizes,
formation control, or robot deployment.

## Optimization integration check

Command: python scripts/check_mappo_update.py

- Device: CUDA.
- One upstream PPO update using synthetic data.
- Finite update outputs: passed.
- Actor parameters changed and remained finite: passed.
- Critic parameters changed and remained finite: passed.

This check does not validate environment rollouts, GAE computation,
reward design, convergence, or formation control.
