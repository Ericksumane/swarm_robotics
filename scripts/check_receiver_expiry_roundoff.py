"""Inclusive age boundaries are stable on the 100 ms simulation clock."""
from swarm_core.communication.receiver import Receiver
from swarm_core.communication.message import StateMessage
for tick in range(5,2401):
 now=tick*.1;created=(tick-5)*.1;r=Receiver(1,.5)
 m=StateMessage(sender_id=0,sequence=tick,generated_at_s=created,x=0.,y=0.,heading=0.,speed=0.,yaw_rate=0.,payload_size_bytes=64)
 r.accept(m,now);assert r.read(0,now)[2]
 assert not r.read(0,now+1e-6)[2]
print('Inclusive expiry survives float roundoff; genuinely stale data still expire: PASSED')
