# Relational actor observation extension

Contract: formation_relational_v4. Previous contracts remain loadable.
The actor receives the same nine local and nine per-neighbor features as v3.
Inside the actor graph, concatenate four deterministic geometry summaries to
the learned masked embedding: mean(e_ij*u_ij) in two body-frame coordinates,
mean(e_ij), and max(abs(e_ij)); e_ij=d_ij/d*-1 and u_ij is the received relative
position normalized to a unit vector. Every aggregation excludes invalid rows.
These are observation features, not velocity commands. A learned neural network
and upstream Gaussian head produce both actions; PPO optimizes all parameters.
No simulator neighbor truth, hidden channel state, expert labels or controller
outputs enter the actor. The summary is permutation invariant.

The previous curriculum experiment did not establish reliable random-heading
performance. This extension tests whether explicit distance/direction correlation
helps the actor, while retaining the same reward and mastery curriculum.
Evaluation uses new development and final-test seed ranges. A single training
seed cannot establish a causal or reliable improvement by itself.
