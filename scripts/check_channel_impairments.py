"""Deterministic channel semantics for burst, outage and bandwidth effects."""
from swarm_core.communication.channel import Channel
from swarm_core.communication.message import StateMessage
def main():
    message=StateMessage(0,0,0.,0.,0.,0.,0.,0.,64)
    outage=Channel(0,1,outages=[[0.,1.]])
    assert not outage.send(message,0.)
    assert outage.send(message,1.)
    burst=Channel(0,1,burst_enter_probability=1.,burst_exit_probability=0.)
    assert not burst.send(message,0.)
    assert not burst.send(message,.1)
    limited=Channel(0,1,bandwidth_bytes_per_s=64.,queue_capacity_bytes=64)
    assert limited.send(message,0.)
    assert not limited.send(message,0.)
    assert limited.receive(.9)==[]
    assert limited.receive(1.)==[(1.,message)]
    assert limited.send(message,1.)
    print('Outage recovery, sustained burst and bounded bandwidth queue: PASSED')
if __name__=='__main__': main()
