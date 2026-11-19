# Fixed nontrivial-start diagnostic

Train a fresh actor for 30000 environment steps, nominal communication,
fixed +/-0.15 m jitter around random triangles and minimum pair error 0.12 m.
Do not increase difficulty by time; first establish acquisition and settling.
All sampled starts must be outside the 0.10 m formation tolerance.
Retain prior v4 reward, std=0.3, gamma=.999, lambda=.99 and entropy=.001
to avoid attributing effects to simultaneous reward/exploration changes.
Save checkpoints near 10000 and 20000 steps plus the final checkpoint.
Select on 10 fixed development episodes, ranking success first, then fewer
collisions, then mean RMSE. Evaluate selected weights on separate seeds.
Compare stationary actions and the previous 30-minute policy using the same
task, seeds and metrics. These comparisons are development evidence, not
hardware validation or a reliability guarantee. No 5 m translation is trained.
