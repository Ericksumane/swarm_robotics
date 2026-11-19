# History-aware collision repair and coordinated startup

## Release status

Experimental three-robot mission bundle:
experiments/runs/history_coordination_001/candidate_stationary_routed.pt.
The previous selected startup and translation registries are preserved.
This release is a candidate for further simulation, not hardware validation.

## Behavior and information boundary

Each robot retains only received packets, predicts peer poses with bounded
history, and sees explicit age/freshness/uncertainty. Main translation experts
still receive their original fresh-only observations. Simulator truth remains
limited to training rewards, centralized critic and evaluation diagnostics.
The Gazebo harness runs the three local computations together for testing;
this is not yet a distributed ROS deployment.

The formation policy uses six independently trained shared MAPPO experts across
the whole mission: fresh formation, history formation, stationary alignment,
original cruise, degraded cruise and approach. Routing is deterministic and
local; these experts are not jointly optimized end to end.

The fresh formation actor is retained exactly. A local hysteresis selector uses
history after four consecutive incomplete fresh observations (0.4 s), returns
after fifteen complete ones (1.5 s), and also uses the collision-trained history
expert when an available estimated peer center is within 0.85 m. The close latch
clears only after both available peers exceed 1.05 m. These thresholds select a
learned actor; they do not calculate a steering command. They are synthetic
development choices and require robot-specific clearance validation.

ALIGN is explicitly rotation in place. Its linear command is constrained to
zero in both training and deployment; the actor chooses yaw. Actual speed still
obeys acceleration limits rather than being teleported to zero. This prevents
small learned linear biases from deforming the formation during negotiation.
The stationary alignment model was warm initialized from the own-state actor
and trained 10000 steps, seed 463, LR 0.00005, gamma 0.99, std 0.02 to 0.01.
Its training completed 77/77 episodes; this is not an evaluation success rate.

## Collision repair objective

Original collision cases 83000003 and 83000005 showed a constant-velocity
predicted overlap 2.3 s and 2.5 s before contact. The existing mean proximity
cost diluted the contribution of one dangerous pair. The new optional training
reward uses the maximum pair cost, a 0.45 m footprint-clearance margin and
weight 90, while preserving the original task and collision termination.

For pair-relative position p and relative velocity v from true training states:

    t_star = clip(-dot(p,v)/max(dot(v,v), 1e-12), 0, H)
    clearance_pred = norm(p + t_star*v) - 2*radius
    risk = max_pairs max(0, (margin - clearance_pred)/margin)^2
    r_closing = -dt * weight * risk

Here H=2 s, margin=0.3 m, weight=20. This is a reward shaping metric, not a
collision-avoidance guarantee or an inference controller. Actor inputs remain
received packets. Perturbed original starts and pre-contact poses were used as
hard training resets; therefore seeds 83000003/5 are development reproductions.

The selected history formation snapshot is step 25088 of close_training, after
a preceding 60000-step collision-reward adaptation. Close-approach exploration
used seed 461, 50000 total steps, LR 0.0001, std 0.18 annealed to 0.02 over
35000 steps, gamma 0.999, lambda 0.99 and unmodified network profiles sampled
with probabilities 0.2/0.2/0.2/0.4. Snapshot 25088 passed 8/8 strict acquisition
tests without collision under nominal and severe communication. Stochastic
training failures are retained in the training logs.

An additional 13-feature collision-input experiment was stopped and is not in
the candidate. Experimental support for that version remains isolated by its
formation_history_risk_v6 contract; the released candidate uses eleven-feature
formation_history_v5. No claim here relies on that unfinished alternative.

## Bounded evidence and future-slot protocol

Each local acquisition router accumulates 2 s of qualifying observed geometry
and low linear speed. Gaps up to 2 s pause accumulation; known bad geometry or
a longer gap resets it. At least two distinct sequence numbers from each peer
are needed. Availability of a prediction never changes receiver freshness.

Own readiness additionally checks local heading and stopped speed, plus peer
estimates, with geometry error <=0.1 m and heading error <=0.05 rad. Readiness
certificates expire after 3 s without qualifying information. Participants
propose a common-clock future slot:

    slot = 5 * ceil((now + 10)/5)

They exchange the maximum proposal through their ordinary impaired messages.
A participant that sees all three offers for the same slot, at least 3 s before
it, broadcasts a committed witness. Other participants can relay/accept that
witness only for a slot they previously offered. This avoids requiring three
identical instantaneous receiver snapshots. Own displacement >0.05 m, invalid
own alignment or an expired slot revokes the local promise. A committed robot
continues stationary alignment until the scheduled departure.

The start proposal and commitment use an additional modeled 8 bytes: candidate
packets cost 72 bytes rather than 64. Loss/delay/jitter/burst/outage/bandwidth
settings are unchanged. These are modeled payload budgets, not measured DDS
wire sizes. The extra load is charged by the channel model.

There is no leader. This protocol assumes synchronized clocks and cooperative
peers. It is not consensus under arbitrary partitions. A robot that receives no
valid commitment before the slot can stay stopped while others depart; expiry,
aborts, clock error and correlated outages require broader validation. Reported
simultaneous starts are measurements of these tests, not a universal guarantee.

## Completion criteria

All profiles require own fixed targets exactly 5 m from measured MOVE starts,
goal error <=0.2 m, speed <=0.03 m/s, yaw rate <=0.05 rad/s and a 2 s hold.
Final pair-error tolerances are explicitly 0.1 m nominal/good, 0.15 m moderate
and 0.2 m severe. The startup requirements stay tighter. The evaluation deadline
is 300 s. Formation error throughout the journey is logged separately; endpoint
completion does not prove perfect continuous geometry.

## Matched full-mission comparison

Four dispersed seeds per profile: 84000000..84000003. These starts were not in
the training reset bank, but were reused during integration development; this
is not reserved confirmatory validation or an estimate of population success.
Reference actors are unchanged and receive matched completion tolerances and
the same 300 s deadline. Candidate payload is larger because of the protocol.

| Profile | Reference completed | Candidate completed | Candidate collisions |
| --- | --- | --- | --- |
| nominal | 4/4 | 4/4 | 0 |
| good | 4/4 | 4/4 | 0 |
| moderate | 4/4 | 4/4 | 0 |
| severe | 0/4 | 4/4 | 0 |

Every candidate start in that sixteen-mission sample had zero measured start
skew. The separate seed 42 Python mission completed in all four profiles with
no collisions. No general 100% success claim follows from these small samples.

The collision reproduction set spans seeds 83000003..83000005, nominal/severe,
including the two formerly colliding starts. See the JSON summary below and
the retained trajectories. A collision-free timeout remains a failed mission.

{
  "nominal": {
    "successes": 2,
    "collisions": 0,
    "started_all": 3
  },
  "severe": {
    "successes": 3,
    "collisions": 0,
    "started_all": 3
  }
}

## Run the candidate live

```bash
cd ~/swarm_robotics
bash scripts/run_live_gazebo.sh --seed 42 --communication severe \
  --time-limit 300 \
  --startup-bundle experiments/runs/history_coordination_001/candidate_stationary_routed.pt
```

Use nominal, good, moderate or severe to select the network profile. The default
launcher without --startup-bundle still uses the previous selected registry.
The GUI run uses live Gazebo wheel physics and continues controlling HOLD;
closing the GUI stops that run. This is not a recorded trajectory replay.


## Final Gazebo wheel-physics check

All four runs used the released candidate, seed 42, actual Gazebo odometry and
300 s deadline. No post-start pose edits or prerecorded trajectory input were
used. Independent audit verified measured MOVE starts, fixed 5 m targets,
fresh odometry (maximum age approximately 0.005 s), and twenty qualifying
stopped feedback frames at completion. All three MOVE timestamps matched in
each run.

| Profile | Completed at | Maximum post-MOVE pair error | Minimum modeled footprint clearance |
| --- | --- | --- | --- |
| nominal | 141.7 s | 0.0927 m | 0.1892 m |
| good | 136.4 s | 0.0929 m | 0.1845 m |
| moderate | 132.0 s | 0.1050 m | 0.1791 m |
| severe | 163.6 s | 0.0885 m | 0.0416 m |

Each profile has one Gazebo run here. These results do not establish a rate
across arbitrary starts or hardware. Severe minimum modeled clearance was
only about 4 cm, so robot-specific footprints and additional safety validation
remain necessary before physical trials.

Across final Python groups (seed 42 in all four profiles, sixteen additional
starts and six collision reproductions), there were 25 completed missions out
of 26 and zero collisions. The remaining failure is nominal seed 83000005:
all robots departed together, but maximum final goal error was about 0.232 m,
above the 0.2 m threshold, and final pair error about 0.118 m. The 300 s timeout
is counted as failure. Its collision was eliminated; arrival is not fully fixed.

All 63 project checks passed, including legacy parity, history exclusion and
expiry, commitment revocation, stationary alignment and local expert routing.
The experimental registry is experiments/selected_history_startup_candidate.json;
the original default registries remain unchanged. Full logs, hashes, trajectories
and measured audits are in experiments/runs/history_coordination_001.
