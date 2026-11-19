"""Verify distance features use received data and preserve loss masking."""
import json
import numpy as np
from swarm_core.simulation import Simulation
from swarm_core.learning.mappo_adapter import make_policy,pack,hidden_inputs
import torch
def main():
 task=json.load(open('configs/tasks/formation_stage1_geometry.json'));comm=json.load(open('configs/communication/nominal.json'))
 sim=Simulation(task,comm);obs=sim.observe_all()
 for row in obs:
  assert row['neighbor_features'].shape==(2,9)
  distance=np.linalg.norm(row['neighbor_features'][:,:2],axis=1)
  np.testing.assert_allclose(row['neighbor_features'][:,7],distance,atol=1e-6)
  np.testing.assert_allclose(row['neighbor_features'][:,8],distance-1,atol=1e-6)
 args,policy,_=make_policy(3,neighbor_dim=9);h,m=hidden_inputs(3,args)
 with torch.no_grad():action,_=policy.act(pack(obs),h,m,deterministic=True)
 assert action.shape==(3,2) and torch.isfinite(action).all()
 blocked=Simulation(task,dict(comm,loss_probability=1.))
 assert all(not x['neighbor_features'].any() for x in blocked.observe_all())
 print('Received geometry features, actor shape and complete-loss exclusion: PASSED')
if __name__=='__main__':main()
