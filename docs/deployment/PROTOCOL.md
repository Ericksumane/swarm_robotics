# Local runtime and peer protocol v1

Each robot owns its odometry, two-peer receiver cache, mission phase, commitment
state, actor routing and terminal stop supervisor. The ROS node talks to its
Python inference process by synchronous versioned JSON-lines. No robot requires
another robot's actor or critic. The critic is not evaluated at inference.

`PeerState.msg` carries protocol_version, mission_id, session_id, sender_id,
sequence, original odometry timestamp, shared frame, x/y/heading/speed/yaw rate,
formation readiness, optional proposed start time, committed witness and fault.
The typed ROS message is converted to the same validated envelope used in tests.
Payload size depends on serialization/strings; the simulator's 72-byte budget
is deliberately not presented as its actual DDS/radio size.

Messages with wrong mission/frame/version, invalid measurements, future/too-old
timestamps, duplicate or decreasing sequence are rejected. A new session ID for
an enrolled peer in the same mission latches a group-stop fault. All nodes must
restart with a fresh mission ID to recover. A timeout/rejected packet never
refreshes the receiver's pose or converts an expired packet into fresh data.

Initial motion waits for both peer sessions to be enrolled. Thereafter local
history may bridge losses. The current guard uses a 5 s peer timeout, 0.3 s
odometry age, 0.5 s control-clock gap and independent wall-clock watchdog in the
ROS wrapper. These are synthetic engineering settings, not verified robot
stopping guarantees. All timestamps assume a common time basis and correct pose
frame registration; no hidden transform or position oracle is used.

Mission phases and history/future-slot coordination follow the tested project
implementation. Source control times are logged; status reception skew is not
the same as physical synchronization error. The protocol is not consensus under
arbitrary partitions. A locally parked robot is not proof that the team has
completed formation/translation; full-team success is measured by evaluation.

The optional terminal stop supervisor requests zero after own goal distance
<=0.19 m and speed/yaw rate below 0.03 m/s / 0.05 rad/s for 2 s. It releases when
own error exceeds 0.2 m. The fixed target is retained. This is an explicit stop
state, not learned steering and not a universal recovery/safety proof. Clock
gaps cannot accumulate settling evidence. Initial mission deadline applies until
the first local arrival; continued HOLD/recovery still checks runtime faults.

The fresh formation expert can be restricted to normalized packet age <=0.1
(0.05 s with the current 0.5 s freshness threshold). Persistent older-but-valid
messages select the prediction/history expert. Receiver freshness semantics do
not change. Inference service errors exit; the ROS parent latches zero commands.


## Phase-specific predictive stop gate

The common release can veto translation using available circular peer estimates.
It retains the actor's yaw request. FORM uses current modeled position bounds
and pauses translation if an estimate is missing. MOVE/HOLD uses point-trajectory
risk for available estimates and leaves unknown-peer behavior to the policy and
5 s peer watchdog. This is an explicitly limited heuristic, not robust safety
under missing observations. Physical protective sensing and stopping remain
unverified. See ARCHITECTURE.md for the trajectory and phase assumptions.

The severe profile's initial mission budget is 420 s; others use 300 s. The ROS
completion_profile parameter selects these task tolerances/deadlines, not actual
network impairment. The Gazebo --time-limit parameter explicitly overrides its
test budget. All versions and failures are retained in the validation report.
