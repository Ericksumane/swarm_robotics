# Formation-stage geometry experiment

Task: formation_stage1_geometry.json. Adds normalized received-neighbor
distance d/d_star and distance error d/d_star-1 to the existing seven rows.
These are computed from delivered measurements, not simulator ground truth.
Invalid rows remain masked and zero. No classical action controller is added.
The actor encoder now accepts a versioned neighbor feature dimension.
The new checkpoint contract is formation_geometry_v2; old checkpoints remain
unchanged and must not be silently interpreted under the new schema.

Fresh MAPPO actor, fixed nontrivial starts, existing proximity-v4 reward.
Initial exploration std=0.1 instead of 0.3 to investigate excessive residual
motion during settling. Gamma=.999, lambda=.99, entropy=.001.
This is a joint development experiment, not an isolated causal ablation.
Training is limited to formation and settling; translation is excluded.
Keep checkpoints near 15000, 30000 and 45000 steps, plus final weights.
Validate before choosing a checkpoint, then use separate comparison seeds.
