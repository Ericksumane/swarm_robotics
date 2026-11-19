# Operational supervisor equations

These supervisors are project code around the frozen MAPPO experts. They are
not rewards, neural weights, online learning or upstream MAPPO equations.

## Terminal stop

For the local own goal g_i frozen when MOVE starts, e_i(t) = ||p_i(t)-g_i||.
The qualifying condition is e_i <= 0.19 m, |v_i| <= 0.03 m/s and
|omega_i| <= 0.05 rad/s, continuously for 2 s. Then a local latch requests
(v_cmd, omega_cmd) = (0,0). It releases when e_i > 0.20 m. Missing or long-gap
control samples do not count as evidence of continuous settling. Implementation:
src/swarm_core/goal_stop.py. A parked actor is a local state, not a claim that
the whole team succeeded.

## Predictive translational veto

The sampled own trajectory uses the acceleration-limited unicycle dynamics in
src/swarm_core/dynamics.py. Peer positions use received constant-twist estimates
from src/swarm_core/communication/prediction.py. At samples k*0.1 s, k=1..10:

    clearance_ij(k) = ||p_i(k)-p_hat_j(k)|| - 2*r - b_j

If any available peer has clearance_ij(k) < 0.03 m, request zero translation
while retaining the actor's yaw request. b_j is the existing history position
bound in FORM and zero during MOVE/HOLD for this release. FORM also vetoes
translation with a missing peer estimate; MOVE/HOLD does not. The local 5 s peer
timeout remains active. Implementation: src/swarm_core/predictive_stop.py.

This sampled one-second heuristic cannot guarantee collision freedom, especially
between samples, with unobserved motion, asymmetric stops, long partitions or
non-circular robots. The model radius is 0.25 m and must not be treated as a
verified Go2, Jackal or RosBot footprint. Real stopping and protective sensing
must be evaluated independently.

## Scheduled start

All local nodes independently choose a common future time slot using the
versioned readiness/proposal/commit protocol. A robot stores only accepted peer
history and its own protocol state. Delayed evidence may defer movement; this
does not establish atomic simultaneous start under arbitrary message loss.
Real start jitter is measured in the ROS smoke report. Implementation:
src/swarm_core/deployment/local_runtime.py and the startup coordinator modules.

The existing project MAPPO/Dec-POMDP documents describe the learned optimization.
These operational extensions must remain separate in any scientific report.
