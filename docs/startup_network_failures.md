# Live startup network failure investigation

User-provided live logs show actual failures, not slow visualization. Nominal
seed 42 first completed at 126.3 s. Moderate never entered MOVE in 240 s and
oscillated between formation/alignment. Severe was closed at 106.5 s with no
formed team, not a completed 240 s test.

Exact replay of channel events matched all logged neighbor masks. Moderate
had 99.15% valid link reads, but alignment/routing did not remain stable.
Severe had only 7.51% valid reads; all robots had zero valid peers in 63.29%
of frames, with a longest simultaneous gap of 3.8 s. The longest consecutive
two-peer-fresh windows were [0.1,0.1,0.2] s, versus the current 2 s startup
requirement. Formation/alignment were trained under nominal communication.
The previously successful severe translation experiment began already formed
and aligned; it did not validate this startup case or Gazebo physics transfer.

## Confirmed numerical defect

Receiver expiry used an exact floating-point age <=0.5 s comparison. Four
severe reads mathematically on the inclusive boundary were rejected by roundoff.
Use a 1 ns absolute numerical tolerance and cap normalized valid age at one.
A genuinely older packet (1 microsecond beyond the boundary) still expires.
This fixes a real defect but is far too small to explain the severe failure.
Loss rates, delay, bandwidth and the 0.5 s freshness requirement are unchanged.

## Bounded alignment architecture trial

Alignment only needs own state and the locally expressed command direction.
Test an explicit own-state actor whose computation graph ignores neighbor
payloads and masks, rather than hoping that nominal-only training learns this
invariance. Input dimensions stay compatible, but the `local_only` architecture
flag is stored in manifests, checkpoints and actor artifacts. Its critic remains
centralized during MAPPO training; inference sees no peer truth. This remains
a shared multi-agent-trained actor, not a scripted yaw controller.

Warm initialize from the selected alignment actor, with fresh optimizers/critic,
20000 steps, seed 458, actor LR 0.0001, gamma 0.99, fixed original GAE lambda,
std 0.08 annealed to 0.01 over 15000 steps, and stronger linear-motion cost 20.
Other alignment reward terms remain unchanged. Select snapshots on eight 82M
development starts, then diagnose Gazebo seed 42 under all four profiles.
These are diagnostic cases, not reserved validation or a guarantee of startup
success. Do not change the selected default solely from these measurements.

No readiness threshold is weakened, no hidden neighbor state is substituted,
and no packet is declared fresh merely to force MOVE. This trial does not solve
severe formation observability or the requirement for sustained fresh peer
information. Those require a separately tested history/estimation and distributed
coordination design, and possibly additional relative sensing on hardware.


## Measured trial results

Candidate: experiments/runs/own_alignment_network_001/candidate_bundle.pt.
Training completed 20000 steps. Both development snapshots passed 8/8 starts;
the selected snapshot was step 20000. All 58 project checks passed.
The actor inference service also preserves the own-state-only architecture.

Headless live Gazebo physics, dispersed seed 42, unchanged communication profiles:

| Profile | Endpoint result | First success | Maximum post-MOVE pair error |
| --- | --- | --- | --- |
| Nominal | Completed | 126.5 s | 0.110643 m |
| Good | Completed | 125.8 s | 0.109663 m |
| Moderate | Completed | 126.8 s | 0.106009 m |
| Severe | Still in FORM at timeout | None in 240 s | Not applicable |

Endpoint completion includes stopping and the final formation hold. None of
the three completed runs satisfies a strict 0.1 m formation-error limit for
the entire MOVE phase. Moderate endpoint errors were 0.1173, 0.1485 and
0.1019 m. There were no collisions in the three completed diagnostic runs.
One seed does not establish a success rate or absence of regression.
The severe trial still failed formation, with final pair error about 1.23 m.

The selected startup and translation registries remain unchanged. To explicitly
try the experimental candidate:

```bash
bash scripts/run_live_gazebo.sh --seed 42 --communication moderate \
  --startup-bundle experiments/runs/own_alignment_network_001/candidate_bundle.pt
```

The default command still uses the selected startup registry. Live logging now
includes valid-neighbor counts, readiness counters/certificates and heading
errors. scripts/audit_live_startup_network.py reproduces historical communication
events and distinguishes original expiry behavior from the corrected boundary.

Next work must address severe startup information loss and readiness coordination
before further training can be interpreted as a solution. The moderate candidate
needs independent seeds and continuous-formation validation before promotion.
