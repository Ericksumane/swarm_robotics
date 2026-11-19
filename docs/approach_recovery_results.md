# Approach recovery results

## Measured conclusion

The frozen two-actor hierarchy improved full-distance endpoint success from
0/40 to 20/40 on matched fresh initial states, with
0 collisions. Continuous formation success remained
0/40. This is a partial research improvement, not completion of all task requirements.

| Fixed 5 m /120 s test | Cruise only | Cruise plus approach |
|---|---:|---:|
| Mission endpoint successes | 0/40 | 20/40 |
| Continuous-formation successes | 0/40 | 0/40 |
| Collisions | 0 | 0 |

Fresh final seeds: 55,000,000..55,000,039. The policies were frozen before this
evaluation. Destination tolerance is 0.2 m; final pair tolerance is 0.1 m;
all speeds must remain <=0.03 m/s and yaw rates <=0.05 rad/s for 2 s.
Median successful completion time: 67.2 s.
Mean fraction of steps inside pair tolerance: 70.9%.
Worst recorded pair error: 0.263 m.

## Failure decomposition

A separate pose audit reconstructed speeds using the exact simulator unicycle
arc and checked the last two seconds. It reproduced every recorded mission
outcome. Point arrival plus stop, excluding geometry, held in
20/40 cases. Remaining mission failures:
goal position 20; stop motion
0; final formation
0.
Failure categories identify the first unmet condition in position, motion,
then geometry order; they are not mutually independent causal findings.
Independently, goal windows passed in 20/40,
stop windows in 22/40, and geometry windows
in 40/40.
This audit does not change scores or tolerances.

## What changed and what did not

The near-goal policy was trained on moving, laterally displaced simulator
encounters, with no teacher actions. Phase-dependent Gaussian exploration
uses std 0.12 during MOVE and 0.01 during HOLD. It changes exploration,
not learned action means. The 100,096-step checkpoint achieved 11/12 near-goal
development successes; the final 150,000-step checkpoint regressed to 0/12.
Checkpoint selection therefore retained the earlier version.

The first full-distance composition entered approach at 0.5 m and scored 0/40
on 53M seeds. Matching entry to the 0.7 m training encounter distance required
preserving own unclipped goal range for the local selector. Actor inputs and
weights were unchanged. The inspected 53M cases became development evidence;
the corrected version was evaluated on the fresh 55M set above.

Both velocity and yaw commands come from frozen MAPPO actors, shared across
robots. Local task-state routing is a project hierarchy, not a claim of joint
end-to-end training by upstream MAPPO. There is no robot leader or centralized
inference input. The original upstream repository remains unchanged.

## Artifacts and limitations

- Configuration and methods: docs/approach_recovery.md.
- Full results and pose audit: experiments/runs/approach_full_validation_v2_001.
- Actor-only bundle: policy_bundle.pt in that directory; export parity passed.
- Translation registry: experiments/selected_translation_candidate.json.
- Gazebo command: bash scripts/view_translation.sh.
- Gazebo demo seed 55000003 is one selected successful episode; aggregate results
  remain 20/40 endpoint and 0/40 continuous formation.

The replay displays Python trajectories on abstract cylinders. It is not a
closed-loop Gazebo physics test. The scene's end pause is display time only.
Initial formation and headings are restricted; poses are ideal in a common
frame. Real hardware interfaces and localization still require confirmation.
The original formation-only candidate is preserved separately.

## Next priority

Reduce residual destination-position errors and improve pair distances during
travel before degraded communication or hardware transfer. Individual 0.2 m goal tolerances alone do
not guarantee 0.1 m pair-distance tolerance; the joint geometry requirement
remains active. Preserve these results as the baseline for that work.
