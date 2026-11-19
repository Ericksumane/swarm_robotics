# Communication Model

## Current implementation

- StateMessage: immutable estimated-state message.
- Channel: one directed link with fixed delay and independent loss.
- Receiver: latest accepted message per sender, with age and validity.

All timestamps currently assume a shared clock basis.
Create new channels and receivers when resetting an episode.

## Directed links

For three robots, use six independent directed links:

0 -> 1
0 -> 2
1 -> 0
1 -> 2
2 -> 0
2 -> 1

Each robot has its own receiver cache.
The Python simulator advances all links; this does not imply a central
communication relay during robot deployment.

## Packet loss

For each transmission, draw u uniformly from [0, 1).

Drop the message if:

u < p_loss

The current model uses independent loss per transmission.
Burst loss and shared network outages are not implemented yet.

## Delivery time

For an accepted transmission:

t_delivery = t_send + delay

Delivery occurs when the simulation polls the channel at or after
t_delivery. No real-time waiting is required.

Generation time and send time are distinct:
t_generation <= t_send <= t_delivery.

## Information age

At observation time t:

age = t - t_generation

A cached message is valid when:

age <= max_age

Missing messages return no state and a false validity flag.
Expired messages remain cached but have a false validity flag.

Observation construction must explicitly handle missing and expired
information. It must never substitute hidden simulator state.

## Ordering

The receiver rejects:
- Messages from itself.
- Duplicate or decreasing sequence numbers.
- Generation timestamps older than the last accepted message.
- Generation timestamps in the future.

Sequence numbers are scoped to one episode.
Deployment restart and session identification remain to be specified.

## Payload size

payload_size_bytes is a declared transmission-budget estimate.
It is not Python memory usage or measured DDS traffic.

The current channel does not enforce bandwidth or queue capacity.

## Verification completed

- Fixed-delay delivery.
- No duplicate delivery from the channel.
- Complete packet loss.
- Independent directed links.
- Missing-message handling.
- Duplicate and out-of-order rejection.
- Information expiration.
- Recovery after receiving a newer message.

## Remaining work

- Integration of all six links.
- Configurable transmission frequency.
- Variable delay and jitter.
- Burst loss.
- Scheduled and shared outages.
- Bandwidth and bounded queues.
- ROS adapters and QoS configuration.
- Clock synchronization and hardware measurements.
## Implemented channel extension

Channel now supports nonnegative uniform extra jitter, a two-state loss
process updated per attempted transmission, configured [start,end) outage
intervals, FIFO serialization at a byte rate and a finite byte capacity.
The queue capacity includes packets currently being serialized, not packets
already serialized and in propagation. Drops are counted by cause.
Accepted packets survive outages that start after admission. This is an
application-channel approximation, not an RF or DDS retransmission model.
All links may use the same outage schedule to model a shared interruption.
Each directed link has its own transmission budget: the model does not yet
represent contention on a single shared Wi-Fi radio channel.
Synthetic good/moderate/severe profiles demonstrate the options, not measured
laboratory network quality. Physical range dependence is not implemented.


## Experimental history-aware startup

The optional history mission separates packet freshness from estimate availability.
Readiness evidence uses multiple packet sequences, explicit expiry and future-slot
commitment propagation; it does not relabel old messages as fresh. This protocol
adds modeled packet bytes and assumes a common clock. See
[History coordination results](history_coordination_results.md) for equations,
observed failures and the limits of partition tolerance. The legacy path remains
the default.
