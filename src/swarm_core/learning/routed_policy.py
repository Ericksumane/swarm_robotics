"""Actor-only inference for a local latched pair of frozen MAPPO policies."""
import numpy as np
import torch
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs
from swarm_core.learning.approach_router import LocalApproachRouter

class RoutedPolicy:
 def __init__(self,bundle):
  if bundle['architecture'] not in ('local_latched_mappo_v1','local_latched_mappo_v2','local_network_routed_mappo_v3') or bundle['contract'] not in ('mission_local_goal_v3','mission_local_goal_time_aligned_v4'):raise ValueError('Unsupported routed policy bundle.')
  self.requires_distance=bundle['architecture'] in ('local_latched_mappo_v2','local_network_routed_mappo_v3')
  self.network_routing=bundle['architecture']=='local_network_routed_mappo_v3'
  if self.network_routing and bundle.get('network_routing')!={'rule':'cruise_degraded_when_any_peer_invalid','required_valid_neighbors':2}:raise ValueError('Unsupported local network routing rule.')
  self.router=LocalApproachRouter(bundle['horizon_m'],bundle['epsilon_m'],bundle.get('observation_horizon_m',.5));self.models={}
  self.width=36
  for name in (('cruise','approach','cruise_degraded') if self.network_routing else ('cruise','approach')):
   item=bundle[name]
   if item['num_agents']!=3 or item['local_dim']!=16 or item['neighbor_dim']!=9:raise ValueError('Unsupported model dimensions.')
   args,policy,_=make_policy(3,local_dim=16,critic_dim=item['critic_dim'],neighbor_dim=9,geometry_summary=item['geometry_summary'],hold_std_scale=item.get('hold_std_scale'),local_only=item.get('local_only',False))
   policy.actor.load_state_dict(item['actor']);policy.actor.eval()
   for p in policy.actor.parameters():p.requires_grad_(False)
   self.models[name]=(policy,*hidden_inputs(1,args))
 def select_model(self,raw,base_mode):
  if not self.network_routing or base_mode!='cruise':return base_mode
  mask=np.asarray(raw)[-2:]
  if not np.all(np.isclose(mask,0)|np.isclose(mask,1)):raise ValueError('Neighbor masks must be binary.')
  return 'cruise_degraded' if np.sum(mask)<2 else 'cruise'
 def reset(self):self.router.reset()
 def act(self,observation,goal_distance_m=None):
  raw=np.asarray(observation,dtype=np.float32)
  if raw.shape!=(self.width,):raise ValueError('Observation width must be 36.')
  if self.requires_distance and goal_distance_m is None:raise ValueError('Own-goal distance metadata is required.')
  mode=self.select_model(raw,self.router.select(raw,goal_distance_m=goal_distance_m));policy,h,m=self.models[mode]
  with torch.no_grad():actions,_=policy.act(raw[None,:],h,m,deterministic=True)
  return actions.numpy()[0],mode


class BatchedRoutedPolicy:
 """Evaluation batching only; each row and each routing state remain local."""
 def __init__(self,bundle):
  self.policy=RoutedPolicy(bundle)
  self.routers=[LocalApproachRouter(bundle['horizon_m'],bundle['epsilon_m'],bundle.get('observation_horizon_m',.5)) for _ in range(3)]
 def reset(self):
  for router in self.routers:router.reset()
 def act(self,observations):
  from swarm_core.learning.mappo_adapter import pack
  if len(observations)!=3:raise ValueError('Exactly three local observations required.')
  raw=pack(observations)
  modes=[self.policy.select_model(row,router.select(row,goal_distance_m=o.get('goal_distance_m'))) for router,row,o in zip(self.routers,raw,observations)]
  actions={}
  with torch.no_grad():
   for name in self.policy.models:
    policy,h,m=self.policy.models[name]
    actions[name],_=policy.act(raw,np.repeat(h,3,axis=0),np.repeat(m,3,axis=0),deterministic=True)
  return torch.stack([actions[mode][i] for i,mode in enumerate(modes)]).numpy(),modes
