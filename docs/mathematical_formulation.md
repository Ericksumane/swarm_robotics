# Cooperative formation learning formulation

## Scope of this original formation-stage document

This section records the original formation-stage training runner, not the
complete later mission runtime. See full_mission.md and
deployment/SUPERVISOR_EQUATIONS.md for the mission and operational extensions.

Formation-and-hold, three synthetic planar agents. Five-meter translation
and distributed phase agreement are not implemented in the training runner.
No online learning occurs during actor inference.

## Dec-POMDP

\(\mathcal M=(I,S,\{A_i\},P,r,\{O_i\},O,\gamma,H)\).
The environment state includes physical states, message queues, receiver
caches, channel RNG state, transmission schedule and readiness counter.
An actor sees its local measurements and valid received neighbor rows.
Messages can be old; local observations are not assumed Markov.
The current actor is feedforward. Recurrence is a future experimental factor.

The policy factorizes as
\(\pi_\theta(\mathbf a|\mathbf o)=\prod_i\pi_\theta(a_i|o_i)\).
Shared weights do not imply shared execution or identical actions.

## Actor extension

Raw neighbor rows and masks are stored in rollouts. Inside the actor,
\(z_i=\sum_j m_{ij}\phi_\theta(f_{ij})/\max(1,\sum_jm_{ij})\).
Append valid fraction and availability, concatenate nine local features,
then apply a two-layer tanh MLP. See learning/mappo_adapter.py.
The unchanged upstream diagonal Gaussian action head produces raw actions.
Dynamics clips and scales them; PPO log probabilities refer to raw samples.
The actor optimizer includes the neighbor encoder.

## Objective and GAE

\(J(\theta)=E[\sum_{t=0}^{H-1}\gamma^t r_t]\).
Let T_t indicate true termination and D_t any episode end/reset:

\(\delta_t=r_t+\gamma(1-T_t)V(s_{t+1})-V(s_t)\).

\(\hat A_t=\delta_t+\gamma\lambda(1-D_t)\hat A_{t+1}\),
\(\hat R_t=\hat A_t+V(s_t)\).

Truncations bootstrap from their final observation, before reset.
Traces never cross episode resets. Rollout boundaries also bootstrap.
Critic input contains physical states, readiness time and agent ID.
It omits hidden channel queues; this is an approximate value representation.

## PPO update

\(\rho_{it}=\exp(\log\pi_\theta(a_{it}|o_{it})-
\log\pi_{old}(a_{it}|o_{it}))\).

\(L_{actor}=-E[\min(\rho\hat A,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)\hat A)]
-\beta E[\mathcal H(\pi)]\).

The official trainer implements this surrogate. For its two-dimensional
Gaussian, upstream action log probabilities are joint log probabilities.
Advantages are standardized over each rollout. Critic updates use upstream
Huber loss with optional value clipping; value normalization is disabled
in this initial runner. Gradients are clipped to norm 0.5.

See docs/reward_design.md for the task reward and docs/dynamics.md for motion.
Default runner gamma=0.999 and lambda=0.95 are development settings.
Gamma's approximate horizon dt/(1-gamma) is 100 s at dt=0.1 s.

## Provenance

Paper: https://arxiv.org/abs/2103.01955
Implementation: https://github.com/marlbenchmark/on-policy
The submodule is unchanged. Set encoder, rollout collection and GAE are
project additions, not a claim of reproducing the paper's benchmarks.
