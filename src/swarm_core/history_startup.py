"""Local history-based acquisition and future-slot startup coordination.

No leader or shared simulator state. This bounded-delay protocol can fail to
start under partitions and is not consensus or a safety proof.
"""
import math
import numpy as np

class EvidenceWindow:
    def __init__(self,required_s=2.,max_gap_s=2.,dt=.1):
        if not np.isfinite([required_s,max_gap_s,dt]).all() or min(required_s,max_gap_s,dt)<=0:raise ValueError('Invalid evidence timing.')
        self.required_s=required_s;self.max_gap_s=max_gap_s;self.dt=dt;self.reset()
    def reset(self):self.duration=0.;self.last=None;self.sequences=[set(),set()]
    def update(self,qualified,known_bad,sequences,now):
        if known_bad or (self.last is not None and now-self.last>self.max_gap_s):self.reset()
        if qualified:
            self.duration+=self.dt;self.last=now
            for seen,seq in zip(self.sequences,sequences):seen.add(int(seq))
        return self.duration>=self.required_s-1e-9 and all(len(seen)>=2 for seen in self.sequences)

def geometry_error(history,desired=1.):
    if not np.all(history['neighbor_mask']==1):return None
    positions=np.vstack([np.zeros(2),history['neighbor_features'][:,:2]])
    return max(abs(np.linalg.norm(positions[i]-positions[j])-desired) for i in range(3) for j in range(i))

class HistoryAcquisitionRouter:
    def __init__(self,desired_distance_m=1.,formation_tolerance_m=.12,max_linear_speed_mps=.03,ready_duration_s=2.,time_step_s=.1,**unused):
        self.desired=desired_distance_m;self.tolerance=formation_tolerance_m;self.speed=max_linear_speed_mps
        self.evidence=EvidenceWindow(ready_duration_s,2.,time_step_s);self.reset()
    def reset(self):self.evidence.reset();self.alignment_latched=False;self.index=0
    def select(self,observation):
        h=observation['formation_history'];error=geometry_error(h,observation.get('desired_pair_distance_m',self.desired));self.index+=1
        now=self.index*self.evidence.dt
        known_bad=error is not None and error>max(.2,self.tolerance)
        if known_bad:self.alignment_latched=False
        slow=abs(observation['local_features'][0])<=self.speed and np.max(np.abs(h['neighbor_features'][:,4]))<=self.speed
        qualified=error is not None and error<=self.tolerance and slow
        ready=self.evidence.update(qualified,known_bad,h['neighbor_sequence'],now)
        if ready:self.alignment_latched=True
        if self.evidence.last is None or now-self.evidence.last>2.:self.alignment_latched=False
        return 'alignment' if self.alignment_latched else 'formation'

class HistoryStartupCoordinator:
    def __init__(self,settings,dt=.1,*,desired_distance_m=1.):
        if not math.isfinite(desired_distance_m) or desired_distance_m<=0:
            raise ValueError('Desired formation distance must be positive and finite.')
        self.desired_distance_m=desired_distance_m
        self.settings=dict(settings);self.evidence=EvidenceWindow(settings.get('required_evidence_s',2.),settings.get('max_evidence_gap_s',2.),dt)
        for key in ('certificate_ttl_s','proposal_lead_s','slot_width_s','geometry_tolerance_m','heading_tolerance_rad'):
            if not math.isfinite(settings[key]) or settings[key]<=0:raise ValueError('Invalid history coordination setting.')
        if settings['proposal_lead_s']<=settings['certificate_ttl_s']:raise ValueError('Proposal lead must exceed certificate lifetime.')
        self.reset()
    def reset(self):self.evidence.reset();self.certificate=False;self.proposal=None;self.last_qualified=None;self.committed=False;self.offered_slots=set();self.offered_positions={}
    def update(self,own,receiver,history,now,direction):
        cfg=self.settings;angle=math.atan2(direction[1],direction[0]);error=geometry_error(history,self.desired_distance_m)
        own_ready=abs(own.speed)<=.03 and abs(own.yaw_rate)<=.05 and abs(math.atan2(math.sin(own.heading-angle),math.cos(own.heading-angle)))<=cfg['heading_tolerance_rad']
        # A complete proposal witness is gossiped. A previously offered slot
        # may be committed using that witness even if direct peer views differ.
        for sender in history['neighbor_ids']:
            message,age,_=receiver.read(sender,now)
            if message is not None and age<=cfg['certificate_ttl_s'] and message.start_committed and message.start_at_s in self.offered_slots and message.start_at_s>=now-1e-8 and own_ready and np.linalg.norm(np.asarray([own.x,own.y])-self.offered_positions[message.start_at_s])<=.05:
                self.committed=True;self.proposal=message.start_at_s;self.certificate=True
        if self.committed:
            if not own_ready or now>self.proposal+self.evidence.dt or np.linalg.norm(np.asarray([own.x,own.y])-self.offered_positions[self.proposal])>.05:
                self.committed=False;self.certificate=False;self.proposal=None;self.evidence.reset()
            else:
                return True,self.proposal,bool(abs(now-self.proposal)<=self.evidence.dt/2+1e-8)
        rows=history['neighbor_features'];headings=np.arctan2(rows[:,2],rows[:,3])+own.heading-angle
        peers_ready=bool(np.all(np.abs(rows[:,4])<=.03) and np.all(np.abs(rows[:,5])<=.05) and np.all(np.abs(np.arctan2(np.sin(headings),np.cos(headings)))<=cfg['heading_tolerance_rad']))
        qualified=own_ready and error is not None and error<=cfg['geometry_tolerance_m'] and peers_ready
        bad=not own_ready or (error is not None and error>.2)
        if bad:self.certificate=False;self.proposal=None
        ready=self.evidence.update(qualified,bad,history['neighbor_sequence'],now)
        if qualified:self.last_qualified=now
        if ready and qualified:self.certificate=True
        if self.last_qualified is None or now-self.last_qualified>cfg['certificate_ttl_s']:
            self.certificate=False;self.proposal=None
        if not self.certificate:return False,None,False
        width=cfg['slot_width_s']
        def next_slot():return math.ceil((now+cfg['proposal_lead_s'])/width-1e-10)*width
        if self.proposal is None:self.proposal=next_slot()
        peer_messages=[]
        for sender in history['neighbor_ids']:
            message,age,_=receiver.read(sender,now)
            if message is None or age>cfg['certificate_ttl_s'] or not message.formation_ready or message.start_at_s is None:
                peer_messages.append(None)
            else:peer_messages.append(message)
        proposals=[m.start_at_s for m in peer_messages if m is not None]
        if proposals:self.proposal=max(self.proposal,max(proposals))
        agreed=all(m is not None and abs(m.start_at_s-self.proposal)<1e-8 for m in peer_messages)
        # Do not move on an expired/late negotiation; negotiate a future slot.
        if now>self.proposal+self.evidence.dt+1e-8 or (now>=self.proposal-1. and not agreed):
            self.proposal=next_slot();agreed=False
        self.offered_slots.add(self.proposal)
        self.offered_positions.setdefault(self.proposal,np.asarray([own.x,own.y]))
        if agreed and self.proposal-now>=cfg.get('minimum_propagation_s',3.):
            self.committed=True
        # Departure happens only from a prior explicit commitment.
        return True,self.proposal,False
