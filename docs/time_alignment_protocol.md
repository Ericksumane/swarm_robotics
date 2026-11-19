# Controlled received-state time alignment

## Evidence and one-variable hypothesis

The frozen registered hierarchy scored 16/16 endpoints with perfect communication
and 5/16 with good communication on matched development seeds 57,000,000..15.
Both scored 0/16 continuous formation. Historical inspection found formation
violations before approach in 38/40 cases, and every robot in the 20 failed
historical missions entered its goal region at least once. Do not attribute
these observations exclusively to communication or claim a population rate.

The sole tested change is predicting each valid received peer pose to receiver
time. Keep actor weights, routing, rates, loss, age feature, TTL, rewards,
initialization, tolerances and episode cutoff fixed. Compare legacy/aligned
observations under nominal/good communication in a 2x2 matched design.

## Estimator and information boundary

For original message age a, payload linear speed v and yaw rate w:
turn=w a, travel=v a sinc(turn/2),
x_hat=x+travel cos(theta+turn/2),
y_hat=y+travel sin(theta+turn/2), theta_hat=wrap(theta+turn).
Here sinc(z)=sin(z)/z with sinc(0)=1. The original timestamp and normalized
age remain unchanged. Expired or missing messages remain masked. Prediction
never reads true peer simulator state. It is an estimator, not a command law.
Zero-age messages return the original pose exactly. Accelerations, clock skew,
measurement noise and long outages limit the constant-twist assumption.

Contract mission_local_goal_time_aligned_v4 retains feature order and dimensions
but versions the changed meaning of peer pose features. Adapted bundles record
the source hash and unchanged actor weights; no retraining is claimed.
Clock synchronization and hardware measurement validation remain necessary.

## Development and final tests

Development uses 16 matched 57M initial states in all four cells. The legacy
cells must reproduce the previous diagnostic outcomes before interpreting
the new results. At nominal zero delay, aligned and legacy observations must
match exactly in unit tests. If promising, freeze the estimator and evaluate
40 new seeds per profile, beginning at 58,000,000. Report endpoint success,
continuous formation, time, clearance and pair errors. Bad/severe tests remain
deferred until the good-communication foundation is satisfactory.
