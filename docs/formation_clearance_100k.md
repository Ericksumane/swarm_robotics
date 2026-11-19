# Formation clearance refinement: 100,000 steps

## Budget and frozen scope

This is a bounded starting budget, not a universal ideal training length.
100,000 three-robot environment steps are allocated to history-formation:
25,000 each in nominal, good, moderate and severe communication. Other five
experts, including the selected arrival refinement, remain unchanged. No
additional steering controller, larger deployment stop margin, changed mission
tolerance or faster motion is introduced. Forward/reverse limits stay
0.33/0.165 m/s. The existing operational veto remains 0.03 m, not 0.20 m.

The target of 0.20 m circular-footprint clearance is a TRAINING preference,
not a guaranteed protective distance. The two user-observed closest pairs
both used history-formation at their minimum clearance. That is why this
experiment updates that expert alone and tests the unchanged fresh-expert
interaction later in full missions/Gazebo.

## Data and training-only objective

45% of resets use rotated/translated/perturbed hard templates; the rest are
uniform dispersed starts in the original spawn area. Templates include the
user's good seed 917443879, nominal seed 613953391, pre-close states and two
separated head-on configurations. The sampler retains at least 0.7 m initial
center separation. It resets velocities to the task's zero initial velocities;
it does not claim exact dynamic replay. No action demonstrations are used.
These templates/cases are development training data, not held-out tests.

FormationSafetyTrainingEnv applies the SAME predictive veto using own state and
accepted receiver history as deployment. The actor keeps local9/history11 inputs.
Ground-truth peer states are used only for reward. Standalone formation training
uses three copies of the history expert; evaluation restores the full local
routing with the unchanged fresh expert and the other stages.

For minimum circular clearance c and target m=0.20 m define
q(c)=clip((m-c)/m,0,1). Additional shared shaping is
r_near=-dt*60*q(c_next)^2 + 8*(q(c_before)^2-q(c_next)^2).
Existing progress/formation/proximity terms remain. Constant-velocity closing
risk weight is 30 with a 2 s horizon and 0.35 m margin; collision/outside-area
event penalties are 1000. This is project-specific shaping, not an upstream
MAPPO equation or policy-invariance guarantee. Actual success criteria and
failure detection in evaluation are unchanged.

Actor warm start is the currently selected arrival-refined bundle. Only its
history-formation actor trains. Actor LR 5e-6, gamma .999, GAE lambda .95,
exploration std .04, PPO clip .1, up to three PPO epochs and four critic-only
warmup updates are used. The critic is freshly initialized. Four profile
trajectories have independent GAE terminal/reset masks. Training sources,
reset settings, budgets and hashes are recorded in
experiments/runs/formation_clearance_100k_001.

## Selection gate fixed before production training

Compare 16 new paired full local-runtime missions (94055000..94055003 per
profile) and the two known close Gazebo cases. Preserve every previously
successful paired mission, observe no sampled collisions, and require no
sampled clearance reduction beyond 0.01 m on new starts. A candidate is only
eligible if both known close cases improve to at least 0.15 m, at least one
reaches 0.20 m, and all known full missions retain completion and terminal hold.
The previously fixed nominal arrival case is rechecked in Gazebo as well.
Successful local-core cases are audited for 60 s of qualifying hold. These
finite gates are not final broad acceptance or hardware collision protection.
A candidate failing the gate is preserved but not selected automatically.
