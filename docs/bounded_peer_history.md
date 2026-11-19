# Bounded peer history experiment

## Status and scope

Experimental acquisition-only contract `formation_history_v5`. Selected startup
and translation bundles are unchanged. Full distributed startup, MOVE/HOLD,
Gazebo and hardware are not validated by acquisition tests. No model promotion
is automatic. Source: src/swarm_core/communication/history.py.

## Information boundary

Only the latest accepted packet per sender and the receiver's own local pose
are used. Missing packets create no synthetic peer. Receiver freshness retains
its original 0.5 s threshold; history availability is a separate mask. No sender
timestamp is refreshed by prediction. A reset clears all cached estimates.

The eleven peer features are: predicted relative x/y in the body frame,
sine/cosine relative heading, reported speed/yaw rate, original age divided by
the freshness threshold, distance/desired, signed distance error/desired,
binary freshness, and modeled position-error bound divided by its cutoff.
Age can exceed one; an available estimate must never be interpreted as fresh.

## Prediction and uncertainty model

Let t be age, v the packet speed, w its yaw rate, and h its heading:

    h_hat = wrap(h + w*t)
    travel = v*t*sinc(w*t/2)
    p_hat = p + travel*[cos(h+w*t/2), sin(h+w*t/2)]

This assumes held reported rates, not held control commands. With configured
linear acceleration A, maximum speed V, yaw-rate magnitude W and initial
position discrepancy b0, the conservative modeled discrepancy is:

    b(t) = b0 + min(A*t^2/2, 2*V*t) + min(V*W*t^2, 2*V*t)

The second term covers speed change and the third covers heading change from
the packet turn rate. It assumes admissible speed, yaw rate, acceleration,
shared time and coordinates. Localization drift, rate errors and acceleration
violations require additional bounds. This is not a measured covariance or
a proof of collision avoidance. Synthetic initial discrepancy b0 is 0.02 m.

Discard the estimate if t>3 s or b(t)>1 m. Thus the effective usable age is
shorter than 3 s for the current limits. Discarded rows are zero with mask zero.
Training can use b(t) to learn caution, but safety under uncertainty is unproven.

## Training and evaluation protocol

Initialize from the selected acquisition actor, extending the first encoder
linear layer from nine to eleven inputs with two zero columns. Preserve all
other actor parameters; start fresh critic and optimizers. Train 50000 steps,
seed 459, LR 0.0001, gamma 0.999, GAE lambda 0.99; exploration std 0.05 to 0.02
over 35000 steps. Network mixture probabilities: nominal/good/moderate/severe
0.2/0.2/0.2/0.4, sampled independently per episode. Profiles are unmodified.
Reward is the existing dispersed acquisition reward, not a steering controller.

Compare identical dispersed starts with the unchanged acquisition actor.
Keep strict acquisition success (0.1 m, stopped for 2 s) and separately report
operational arrival (0.2 m, stopped for 2 s). Operational arrival is a diagnostic
milestone, not a full mission success and does not erase subsequent collisions.
Report collisions, final geometry and information availability. Development
results are not reserved validation. Readiness logic is unchanged: the existing
requirement for sustained fresh peers remains a separate severe-network blocker.


## Completed results

Run: experiments/runs/bounded_history_001. The 50000-step training completed
in 589.6 s: 84 training episodes (63 success, 15 timeout, 6 collision). Training
outcomes include mixed communication and near-formation resets; they are not
evaluation success rates. All 59 project checks and actor export parity passed.

Matched development seeds 83000000..83000007, fully dispersed initial states,
120 s limit, identical 0.1 m formation tolerance and stopped hold for 2 s:

| Profile | Original success | History success | Original collisions | History collisions |
| --- | --- | --- | --- | --- |
| Nominal | 6/8 | 6/8 | 2 | 2 |
| Good | 7/8 | 6/8 | 1 | 2 |
| Moderate | 7/8 | 6/8 | 1 | 2 |
| Severe | 0/8 | 6/8 | 0 | 2 |

The 0.2 m operational-arrival metric did not recover extra successes in these
final tests. This is acquisition only in Python dynamics, not a completed 5 m
mission, not Gazebo and not hardware. Eight starts are insufficient for a
reliable population rate. No reserved validation claim is made.

Decision: do not promote. History clearly restores useful information under
severe loss in this sample, but adds one failed/collision episode in good and
moderate conditions. The selected startup and translation registries remain
unchanged. The own-state alignment candidate that worked in live moderate
communication remains available separately, with its original command.

The intermediate checkpoint at 25088 steps produced the same four success
counts. Its selection is not a way to eliminate the observed regression.
The initial baseline comparison accidentally used a legacy 0.05 m task and was
excluded (baseline_invalid_mismatched_tolerance.json); the table uses the fully
repeated comparison with identical tasks. No mixed-threshold result is used.

Next engineering work: investigate collision trajectories and preserve the
fresh-message behavior when adding history; then test history-aware local
acquisition/alignment routing and a loss-tolerant distributed readiness protocol.
Current readiness remains unchanged and can still block MOVE in severe loss.
Do not solve that by relabeling estimates as fresh or silently reducing losses.

Reproduce the isolated comparison:

```bash
python scripts/check_peer_history.py
python scripts/evaluate_peer_history.py \
  --checkpoint experiments/runs/bounded_history_001/training/checkpoint_0050000.pt \
  --label repeated_history --episodes 8
```

The exported experimental actor includes the history construction settings at
experiments/runs/bounded_history_001/export_50k/actor.pt. It must not be loaded
as a nine-feature acquisition actor in the current live mission bundle.
