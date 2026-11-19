# Severe communication arrival recovery

The selected three-expert composition improved severe communication without
observed nominal/good regressions on the matched reserved cases. The dispersed
formation and complete-chain validation did not pass; see dispersed_validation_results.md.

## Reserved matched results

40 fixed starts 71000000 through 71000039 per profile. Checkpoint selected on
69M development starts before these tests; the original reference was then
evaluated on the same 71M starts to distinguish existing failures from regression.

| Communication | Original endpoint / strict | Candidate endpoint / strict | Candidate collisions | Rescued / regressed |
|---|---:|---:|---:|---:|
| nominal | 38/40 / 38/40 | 38/40 / 38/40 | 0 | 0 / 0 |
| good | 40/40 / 40/40 | 40/40 / 40/40 | 0 | 0 / 0 |
| moderate | 40/40 / 40/40 | 40/40 / 40/40 | 0 | 0 / 0 |
| severe | 0/40 / 0/40 | 40/40 / 40/40 | 0 | 40 / 0 |

The earlier 40/40 nominal result used a different 61M set. The new nominal
set contains two failures shared by both models. Do not present 40/40 as a
general guarantee or hide these failures. Physical tolerances and the 120 s
limit were not relaxed. Strict success requires every pair-distance error
<=0.1 m at every frame plus final goal error <=0.2 m, linear speed <=0.03 m/s
and yaw speed <=0.05 rad/s for 2 s. Localization and robot dynamics remain synthetic.

## Why the first fixes were rejected

Approach-only continuation: 40000 steps, mixed networks, original cruise frozen.
It retained 40/40 nominal, good and moderate on reserved 66M starts but remained
0/40 severe. Captured severe episodes already had mean lateral error 0.373 m
at approach entry, versus 0.095 m with good communication. Near-goal training
alone targeted the wrong source of the displacement. These measurements are
from separate training-capture seeds, not the reserved selection cases.

Cruise-only mixed-network continuation: 20000 steps. Its first snapshot achieved
4/4 strict severe development successes but 0/4 nominal and good, so a direct
weight replacement was rejected. The fixed local composition retains the
original cruise and approach weights and adds the 10240-step degraded expert.

## Shared local policy and mathematics

Every robot loads the same three offline-trained MAPPO actors. Let m_ij(t)
be the validity mask from its own receiver for peer j, M_i(t)=product_j m_ij(t),
and L_i(t) the existing local approach latch (own range <0.7 m or local HOLD).

a_i(t) = pi_A(o_i) if L_i(t)=1;
otherwise pi_C(o_i) if M_i(t)=1, otherwise pi_D(o_i).

C and A are the unchanged original cruise and approach actors; D is the
new degraded-cruise actor. The availability rule only affects cruise. The
router has no network-profile label, global quality score, neighbor truth or
central command. It uses the existing binary expiration masks. No analytical
motion controller changes the neural actions, and inference does not adapt weights.

The gating rule is hand-defined. The experts were trained independently with
unchanged upstream MAPPO PPO updates; this is not a jointly trained gating
network or a single end-to-end-trained actor. This extra architecture complexity
is a tradeoff and should be compared with later joint-policy training. Existing
actor/critic/GAE/PPO equations remain documented in the project; mixture and
routing are project-specific extensions described in arrival_and_acquisition_protocol.md.

## Verification and artifacts

All 50 project checks passed. Additional integration verification passed for
the one-robot JSON-lines actor service in all three modes, unchanged C/A weights,
and exact reproduction of the recorded severe demo. All-valid actions match the
legacy policy; separate-robot and batched decisions match. The selected registry
and bundle hash are in experiments/selected_translation_candidate.json; previous
registry is retained in experiments/references/translation_before_network_recovery.json.

Gazebo view: `bash scripts/view_translation.sh`. The demo now uses a successful
severe episode, with target markers and a per-step quality_trace.json. This is
recorded Python pose playback, not Gazebo closed-loop physics or hardware validation.
No desktop UI was controlled to prepare it.

The next unresolved task is broad-position acquisition and an appropriately
trained startup/alignment transition, not further increasing the mission timeout.
