"""Check explicit stale status, bounded prediction, expiry, reset and gradients."""
import math
import numpy as np
import torch
torch.set_num_threads(1)
from swarm_core.communication.receiver import Receiver
from swarm_core.communication.message import StateMessage
from swarm_core.communication.history import build_history_observation
from swarm_core.dynamics import RobotState
from swarm_core.formation_env import FormationEnv
from swarm_core.learning.mappo_adapter import make_policy,pack,critic_input,hidden_inputs
import json

settings=json.load(open("configs/tasks/dispersed_history_training.json"))["peer_history"]
r=Receiver(0,.5);own=RobotState()
missing=build_history_observation(own,r,[1,2],0.,1.,settings)
assert not missing["neighbor_mask"].any()
message=StateMessage(sender_id=1,sequence=1,generated_at_s=0.,x=1.,y=0.,heading=0.,speed=.2,yaw_rate=0.,payload_size_bytes=64)
assert r.accept(message,0.)
fresh=build_history_observation(own,r,[1,2],.4,1.,settings)
np.testing.assert_allclose(fresh["neighbor_features"][0,0],1.08)
assert fresh["neighbor_fresh_mask"][0]==1
stale=build_history_observation(own,r,[1,2],.8,1.,settings)
assert stale["neighbor_mask"][0]==1 and stale["neighbor_fresh_mask"][0]==0
assert stale["neighbor_features"][0,6]>1 and stale["neighbor_features"][0,9]==0
assert stale["neighbor_position_error_bound_m"][0]>fresh["neighbor_position_error_bound_m"][0]
assert not r.read(1,.8)[2], "History must not change receiver freshness."
expired=build_history_observation(own,r,[1,2],3.1,1.,settings)
assert not expired["neighbor_mask"].any() and not expired["neighbor_features"].any()
task=json.load(open("configs/tasks/dispersed_history_training.json"));comm=json.load(open("configs/communication/severe.json"));reward=json.load(open("configs/training/dispersed_acquisition_reward.json"))
env=FormationEnv(task,comm,reward);obs,_=env.reset(42)
assert pack(obs).shape==(3,33)
args,policy,trainer=make_policy(3,local_dim=9,neighbor_dim=11,critic_dim=23,geometry_summary=True)
h,m=hidden_inputs(3,args)
for _ in range(15):obs,*_=env.step(np.zeros((3,2)))
raw=torch.tensor(pack(obs),requires_grad=True)
policy.actor.base(raw).square().sum().backward()
assert torch.isfinite(raw.grad).all()
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.actor.base.neighbors.parameters())
# A change to untransmitted peer estimates cannot enter robot 0's history.
before_peer=env.simulation.observe(0)["neighbor_features"].copy()
from dataclasses import replace
env.simulation.estimates[1]=replace(env.simulation.estimates[1],x=999.,y=999.)
np.testing.assert_array_equal(before_peer,env.simulation.observe(0)["neighbor_features"])
before=pack(env.reset(42)[0]);env.step(np.zeros((3,2)));after=pack(env.reset(42)[0])
np.testing.assert_array_equal(before,after)
assert all(not receiver._latest for receiver in env.simulation.network.receivers.values())
print("Missing/fresh/stale/expired history, uncertainty and reset: PASSED")
print("History actor dimensions and encoder gradients: PASSED")
