"""Check relative-risk geometry from independently constructed packets."""
import math,json
import numpy as np
import torch
from swarm_core.communication.history import build_history_observation
from swarm_core.communication.receiver import Receiver
from swarm_core.communication.message import StateMessage
from swarm_core.dynamics import RobotState
from swarm_core.learning.mappo_adapter import make_policy

torch.set_num_threads(1)
settings=json.load(open('configs/tasks/history_risk_training.json'))['peer_history']
receiver=Receiver(0,.5)
def message(sequence,heading):
    return StateMessage(sender_id=1,sequence=sequence,generated_at_s=0.,x=1.,y=0.,heading=heading,speed=.2,yaw_rate=0.,payload_size_bytes=64)
receiver.accept(message(1,math.pi),0.)
approach=build_history_observation(RobotState(speed=.2),receiver,[1,2],0.,1.,settings)
assert approach['neighbor_features'].shape==(2,13)
np.testing.assert_allclose(approach['neighbor_features'][0,11:],[.4,-.3],atol=1e-6)
assert not approach['neighbor_features'][1].any()
receiver.accept(message(2,0.),0.)
parallel=build_history_observation(RobotState(speed=.2),receiver,[1,2],0.,1.,settings)
np.testing.assert_allclose(parallel['neighbor_features'][0,11:],[0.,.5],atol=1e-6)
assert not receiver.read(1,.8)[2]
stale=build_history_observation(RobotState(speed=.2),receiver,[1,2],.8,1.,settings)
assert stale['neighbor_mask'][0]==1 and stale['neighbor_fresh_mask'][0]==0
args,policy,_=make_policy(3,local_dim=9,neighbor_dim=13,critic_dim=23,geometry_summary=True)
raw=torch.randn(3,37,requires_grad=True)
with torch.no_grad():raw[:,-2:]=1.
policy.actor.base(raw).square().sum().backward()
assert torch.isfinite(raw.grad).all() and raw.grad[:,9+11:9+13].abs().sum()>0
print('Received-only closing speed, closest clearance, stale labeling and gradients: PASSED')
