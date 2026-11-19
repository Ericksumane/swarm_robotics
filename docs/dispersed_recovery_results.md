# Dispersed formation and learned alignment progress

The acquisition component improved substantially, but the complete dispersed
mission is not validated. The earlier severe-communication translation policy
and its registry were preserved. Startup components were evaluated only under
nominal and good communication, with ideal shared localization.

## Formation acquisition: paired reserved comparison

40 fresh 73M starts per profile, with uniform positions in [-1.5,1.5] m per
axis, separation >=0.7 m, independent uniform headings and zero speed. The
physical acceptance criterion remains pair error <=0.1 m and original speed
bounds continuously for 2 s, within 120 s.

| Profile | Original success / collisions | First recovery success / collisions |
|---|---:|---:|
| nominal | 20/40 / 2 | 38/40 / 1 |
| good | 21/40 / 2 | 38/40 / 1 |

An unchanged first-recovery actor was then tested on 40 additional fresh
78M starts per profile: 40/40 successes and zero collisions in each. Across
the two disjoint cohorts, this actor achieved 78/80 successes with one collision
per profile. The replication does not erase the collision in the first cohort.
It is not a guarantee for arbitrary placements or hardware.

## Alignment component

The separately learned MAPPO alignment actor achieved 40/40 successes and zero
collisions per nominal/good profile on fresh 74M starts. This test began with
valid geometry and random headings. Success requires heading error <=0.04 rad,
geometry error <=0.1 m and original speed bounds for 2 s. It is not itself a
complete formation-from-dispersion or translation mission.

## Complete-chain integration

20 fresh 75M starts per profile. The same cases were tested at two separately
labelled time limits. These are paired timing diagnostics, not independent
samples to pool. Strict success requires complete formation/alignment/5 m/stop
and no pair error above 0.1 m after all agents start MOVE.

| Profile | Limit | Endpoint success | Strict post-MOVE success | Collisions |
|---|---:|---:|---:|---:|
| nominal | 120 s | 8/20 | 7/20 | 0 |
| good | 120 s | 8/20 | 7/20 | 0 |
| nominal | 240 s | 18/20 | 16/20 | 0 |
| good | 240 s | 19/20 | 17/20 | 0 |

More time allows startup plus translation to finish in additional cases, but
does not establish continuous formation: some completed missions still exceed
the pair-distance tolerance after MOVE. The original 120 s results must not
be replaced by the 240 s diagnostic. No positions were edited in these rollouts.

## Rejected follow-up

Fresh 76M failure mining and close head-on training templates were used to
train a 0.05 m precision/safety variant for 30000 steps. Development precision
improved, but the 78M original-criterion paired benchmark regressed:

| Profile | Preserved first recovery | Rejected refinement |
|---|---:|---:|
| nominal | 40/40, 0 collisions | 33/40, 4 collisions |
| good | 40/40, 0 collisions | 34/40, 3 collisions |

The refinement and its tighter-startup integration were not adopted. Its raw
diagnostics remain archived, including any completed rejected integration runs.
The validation runner now explicitly skips promotion/integration on an observed
paired collision regression. This negative result illustrates why more training,
stronger penalties or a tighter threshold alone cannot be assumed to improve safety.

## Architecture and verification

The optional startup bundle contains five shared frozen learned actors: formation,
alignment, original cruise, degraded cruise, and approach. Each robot selects
using its own phase, delivered-neighbor masks/geometry and local readiness
counters. The startup heading readiness threshold is 0.05 rad. The preserved
candidate uses the original 0.1 m startup geometry criterion, not the rejected
0.05 m margin. Actions come from MAPPO actors; there is no scripted turning
controller, online weight update, central leader or pose correction. Experts
and gating were not trained jointly end-to-end. The alignment reward and all
training settings are documented in dispersed_recovery_protocol.md.

All 54 project checks passed, including local routing isolation, learned-action
parity, readiness conditions and legacy mission behavior. The startup bundle
loads and reproduces the selected recorded successful rollout exactly. Main
translation weights match the previously evaluated artifact hash.

Research startup registry: experiments/selected_startup_candidate.json.
It is separate from experiments/selected_translation_candidate.json. Neither
the startup bundle nor its successful replay establishes hardware readiness.

## View the recorded startup example

```bash
cd ~/swarm_robotics
bash scripts/view_startup.sh
```

This replay shows a successful 120 s good-communication case, labelled with
the full evaluation counts. It displays recorded Python poses in Gazebo, not
Gazebo closed-loop physics. Target markers show goals frozen at the recorded
MOVE transition, as visual annotations. The preceding formation displacement
does not count toward the commanded 5 m translation.

## Remaining work

Acquisition still has an observed rare collision. Some startups are slow, some
alignment/geometry transitions fail, and some translated formations exceed the
distance tolerance. Further work should target these failures with fresh
training/development/reserved sets, rather than repeatedly reusing reserved
tests or increasing the timeout until a preferred score appears. Startup under
moderate/severe communication, measurement drift, real robot dynamics and lab
integration remain unvalidated.
