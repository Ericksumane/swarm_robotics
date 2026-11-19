"""Line-delimited local inference service; no training or critic calls."""
import json,sys
import torch
from swarm_core.learning.mappo_adapter import CONTRACT,make_policy,hidden_inputs
import numpy as np
torch.set_num_threads(1)
checkpoint=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
if checkpoint['contract'] not in (CONTRACT,'formation_history_risk_v6','formation_history_v5','mission_local16_v1','formation_geometry_v2','formation_geometry_critic_v3','formation_relational_v4','mission_relational_v2','mission_local_goal_v3','mission_local_goal_time_aligned_v4'): raise ValueError('Contract mismatch')
args,policy,_=make_policy(checkpoint['num_agents'],local_dim=checkpoint.get('local_dim',9),critic_dim=checkpoint.get('critic_dim'),neighbor_dim=checkpoint.get('neighbor_dim',7),geometry_summary=checkpoint.get('geometry_summary',False),hold_std_scale=checkpoint.get('hold_std_scale'),local_only=checkpoint.get('local_only',False))
policy.actor.load_state_dict(checkpoint['actor']);policy.actor.eval()
h,m=hidden_inputs(1,args)
for line in sys.stdin:
 try:
  raw=np.asarray(json.loads(line)['observation'],dtype=np.float32).reshape(1,-1)
  with torch.no_grad(): action,_=policy.act(raw,h,m,deterministic=True)
  response={'action':action[0].tolist()}
 except Exception as error: response={'error':str(error)}
 print(json.dumps(response),flush=True)
