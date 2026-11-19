"""Verify stationary ALIGN mapping and local expert hysteresis."""
import copy,json
import numpy as np
import torch
from swarm_core.alignment_env import AlignmentEnv
from swarm_core.learning.formation_expert_router import LocalFormationExpertRouter
from swarm_core.learning.staged_policy import BatchedMissionPolicy

torch.set_num_threads(1)
task=json.load(open('configs/tasks/alignment_stationary_v3.json'));comm=json.load(open('configs/communication/nominal.json'));reward=json.load(open('configs/training/alignment_reward.json'))
env=AlignmentEnv(task,comm,reward);env.reset(1)
positions=np.asarray([[s.x,s.y] for s in env.simulation.states])
for _ in range(20):env.step(np.tile([1.,.3],(3,1)))
np.testing.assert_array_equal(positions,[[s.x,s.y] for s in env.simulation.states])
assert all(s.speed==0 for s in env.simulation.states)
router=LocalFormationExpertRouter()
history=dict(neighbor_features=np.array([[1.,0.]+[0.]*9,[.5,np.sqrt(3)/2]+[0.]*9]),neighbor_mask=np.ones(2))
o=dict(neighbor_mask=np.ones(2),formation_history=history)
assert router.select(o)=='formation_fresh'
lost=copy.deepcopy(o);lost['neighbor_mask'][:]=0
for _ in range(4):name=router.select(lost)
assert name=='formation'
for _ in range(14):assert router.select(o)=='formation'
assert router.select(o)=='formation_fresh'
close=copy.deepcopy(o);close['formation_history']['neighbor_features'][0,0]=.8
assert router.select(close)=='formation'
router.reset();assert router.select(o)=='formation_fresh'
print('Stationary alignment preserves positions and local expert hysteresis resets: PASSED')
