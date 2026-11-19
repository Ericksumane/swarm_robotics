"""Directed seeded application channel; not a radio-propagation simulator."""
import heapq
import math
from itertools import count
import numpy as np
from .message import StateMessage

class Channel:
    def __init__(self, sender_id, receiver_id, delay_s=0., loss_probability=0.,
                 seed=42, jitter_s=0., bandwidth_bytes_per_s=None,
                 queue_capacity_bytes=65536, burst_enter_probability=0.,
                 burst_exit_probability=1., bad_state_loss_probability=1.,
                 outages=None):
        if any(type(i) is not int or i<0 for i in (sender_id,receiver_id)) or sender_id==receiver_id:
            raise ValueError('Invalid directed robot IDs.')
        for value in (delay_s,jitter_s):
            if not math.isfinite(value) or value<0: raise ValueError('Invalid delay.')
        for value in (loss_probability,burst_enter_probability,burst_exit_probability,bad_state_loss_probability):
            if not math.isfinite(value) or not 0<=value<=1: raise ValueError('Invalid probability.')
        if bandwidth_bytes_per_s is not None and (not math.isfinite(bandwidth_bytes_per_s) or bandwidth_bytes_per_s<=0):
            raise ValueError('Bandwidth must be positive.')
        if type(queue_capacity_bytes) is not int or queue_capacity_bytes<1:
            raise ValueError('Queue capacity must be a positive integer.')
        self.outages=[] if outages is None else [tuple(x) for x in outages]
        for start,end in self.outages:
            if not all(math.isfinite(x) for x in (start,end)) or start<0 or end<=start:
                raise ValueError('Invalid outage interval.')
        self.sender_id,self.receiver_id=sender_id,receiver_id
        self.delay_s,self.jitter_s=delay_s,jitter_s
        self.loss_probability=loss_probability
        self.bandwidth=bandwidth_bytes_per_s
        self.capacity=queue_capacity_bytes
        self.burst_enter=burst_enter_probability
        self.burst_exit=burst_exit_probability
        self.bad_loss=bad_state_loss_probability
        self.bad=False
        self.rng=np.random.default_rng(seed)
        self._queue=[]; self._order=count(); self._time=0.; self._free=0.
        self._transmissions=[]
        self.sent=self.dropped=self.delivered=0
        self.drop_reasons=dict(outage=0,loss=0,queue=0)

    def _advance(self,now_s):
        if not math.isfinite(now_s) or now_s<self._time:
            raise ValueError('Time must be finite and nondecreasing.')
        self._time=now_s
        self._transmissions=[(end,size) for end,size in self._transmissions if end>now_s]

    def _drop(self,reason):
        self.dropped+=1; self.drop_reasons[reason]+=1
        return False

    def send(self,message: StateMessage,now_s):
        if message.sender_id!=self.sender_id or message.generated_at_s>now_s:
            raise ValueError('Invalid sender or future message.')
        self._advance(now_s); self.sent+=1
        if any(start<=now_s<end for start,end in self.outages):
            return self._drop('outage')
        if self.bad:
            if self.rng.random()<self.burst_exit: self.bad=False
        elif self.rng.random()<self.burst_enter: self.bad=True
        probability=self.bad_loss if self.bad else self.loss_probability
        if self.rng.random()<probability: return self._drop('loss')
        finish=now_s
        if self.bandwidth is not None:
            if sum(size for _,size in self._transmissions)+message.payload_size_bytes>self.capacity:
                return self._drop('queue')
            finish=max(now_s,self._free)+message.payload_size_bytes/self.bandwidth
            self._free=finish
            self._transmissions.append((finish,message.payload_size_bytes))
        delay=self.delay_s + (self.rng.uniform(0,self.jitter_s) if self.jitter_s else 0.)
        # Accepted packets survive a later outage: this is an application model.
        heapq.heappush(self._queue,(finish+delay,next(self._order),message))
        return True

    def receive(self,now_s):
        self._advance(now_s)
        received=[]
        while self._queue and self._queue[0][0]<=now_s:
            delivery,_,message=heapq.heappop(self._queue)
            received.append((delivery,message))
        self.delivered+=len(received)
        return received
