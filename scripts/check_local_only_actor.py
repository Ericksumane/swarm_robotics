"""Own-state actor outputs and gradients cannot depend on neighbor messages."""
import numpy as np
import torch
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs
torch.set_num_threads(1);torch.manual_seed(3)
args,p,_=make_policy(3,local_dim=16,critic_dim=39,neighbor_dim=9,geometry_summary=True,local_only=True)
raw=np.random.default_rng(4).normal(size=(3,36)).astype(np.float32);raw[:,-2:]=1;other=raw.copy();other[:,16:34]=100.;other[:,-2:]=0
h,m=hidden_inputs(3,args)
with torch.no_grad():a,_=p.act(raw,h,m,deterministic=True);b,_=p.act(other,h,m,deterministic=True)
torch.testing.assert_close(a,b,rtol=0,atol=0)
tensor=torch.tensor(raw,requires_grad=True);p.actor.base(tensor).sum().backward();assert torch.count_nonzero(tensor.grad[:,16:])==0 and torch.count_nonzero(tensor.grad[:,:16])>0
print('Own-state policy is exactly invariant to neighbor payloads/masks: PASSED')
