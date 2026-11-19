# Common runtime delivery validation

## Candidate and scope

Release: common_shadow_runtime_001. Policy SHA-256: `e3c5084a429325f3fdf1bedadee9e725672514a868f8ca37056836a8304c4740`.
Source policy: experiments/runs/delivery_completion_001/candidate_runtime_v5.pt.
The delivery registry and manifest pin this artifact. Upstream MAPPO remains
unchanged at de66d7a4b23fac2513f56f96f73b3f5cb96695ac.

This delivery completes the hardware-independent common shadow runtime, not
hardware implementation or final statistical policy acceptance. No neural
weights were trained during this revision. The six existing shared MAPPO experts
remain fixed. Changes include age-aware expert routing, local memory/phase
execution, a terminal stop latch and a phase-specific predictive translation
veto. These deterministic operational extensions are explicitly not learned
control. See deployment/ARCHITECTURE.md and SUPERVISOR_EQUATIONS.md.

## Complete movement and continuous stopping

| Profile | Development local-core missions | New frozen-candidate missions | Gazebo first completion (s) | Gazebo subsequent hold (s) | Gazebo largest final goal error (m) |
| --- | --- | --- | --- | --- | --- |
| nominal | 4/4 | 2/2 | 141.6 | 218.4 | 0.181 |
| good | 4/4 | 2/2 | 130.7 | 229.3 | 0.139 |
| moderate | 4/4 | 2/2 | 140.7 | 219.3 | 0.133 |
| severe | 4/4 | 2/2 | 203.2 | 276.8 | 0.109 |

Local-core evaluation uses three independent mission runtimes and a synthetic
message network over a Python physical plant. Development seeds are
85000000..85000003; new seeds are 87000000..87000001 per profile. All 24 runs
included 60 seconds after initial completion. The continuous trace auditor found
zero terminal goal/formation/speed violations in these windows and no observed
collisions. These are 24 finite experiments, not a general 100% success claim.
The eight new runs were not used to tune this frozen candidate.

Gazebo uses synthetic three-robot wheel dynamics, seed 42 per profile, with live
feedback rather than pose replay. All four completed; none collided in this
sample. Continuous hold was audited through 360 s (480 s severe), with no
terminal tolerance violations. The recorded reason `duration_limit` is expected
because `--keep-controlling` intentionally continues after first success.

The goal is 5 m from each robot's frozen MOVE start position, not from its random
episode position. Formation acquisition can move robots beforehand. Goal
tolerance is 0.2 m; final pairwise formation tolerance is 0.1/0.1/0.15/0.2 m for
nominal/good/moderate/severe. Initial mission deadline is 300 s except severe
420 s. One development severe completion took 332.8 s; explicitly increasing its
budget is part of this result, not evidence of a faster learned policy.

## Integration and clean installation

- All 68 project checks passed after runtime changes.
- Batched/local actor parity passed over nominal and severe complete episodes;
  maximum action difference was approximately 1.19e-7. Versioned local service
  reset and checksum rejection passed.
- A newly extracted x86_64 Ubuntu 24.04/ROS Jazzy package installed successfully
  into its own Python 3.10 CPU environment with PyTorch 2.5.1+cpu. Dependency
  checks, model loading, both ROS package builds and shadow preflight passed.
- Three independent ROS processes from that installed CPU package passed typed
  peer/start/fault tests with stationary odometry. Removing one odometry stream
  caused all three to latch zero shadow commands. This is a transport/runtime
  smoke test, not a physical trajectory or three-host network test.

The ROS source start skew in that loaded test was 0.0529 s;
maximum per-tick compute was 0.0829 s. These are measured samples,
not hard real-time guarantees. Real clocks, hosts and latency require validation.

## Failed variants retained

The earlier age-qualified v2 local-core evaluation collided in severe seed
85000001. A more conservative v3 stop gate avoided collisions in that sample but
two severe missions hit the deadline. The v4 local-core sample completed 16/16,
yet its seed-42 severe Gazebo test failed to leave FORM before 300 s. These
counterexamples motivated the explicit v5 phase tradeoff and longer severe
budget. Their logs remain under experiments/runs/delivery_completion_001.
Do not combine earlier variant results into v5 evidence.

## Reproduction and evidence

Run `python scripts/check_all.py` in the development environment. Use
`scripts/validate_local_runtime.py` with the pinned bundle for independent local
core tests, and `scripts/audit_runtime_hold.py --root
experiments/runs/delivery_completion_001` for the saved terminal traces. The
root audit contains runtime_v5_validation, runtime_v5_independent and four
gazebo_v5_hold_PROFILE directories, plus ros_cpu_release_smoke/result.json and
local_inference_parity_v5.json. Full training environments/logs are not included
in the minimal runtime archive. The archive carries the actual model weights,
common source, English instructions, manifest and SHA256SUMS.

To open the current candidate in the development Gazebo GUI:

```bash
cd ~/swarm_robotics
bash scripts/run_delivery_gazebo.sh good --seed 42
```

Replace good with nominal, moderate or severe. This verifies the candidate hash
and selects its budget explicitly. It does not change historical default
registries. See deployment/README.md for the separate runtime installation.

## Unfinished acceptance and hardware work

The proposed 100 episodes per profile benchmark has not been completed and its
acceptance flag remains false. The current small sample is a useful baseline,
not proof that more diverse starts, partitions or heterogeneous dynamics work.
Real robot profiles, physical footprints/limits, onboard architecture, vendor
adapters, sensors, common localization, clock alignment and independent stopping
remain pending. The predictive veto is a sampled heuristic and cannot guarantee
collision freedom. No vendor motor interface is connected by this release;
hardware preflight intentionally fails. See deployment/HARDWARE_PENDING.md.
