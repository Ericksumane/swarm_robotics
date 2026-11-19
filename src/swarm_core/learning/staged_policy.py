"""Local FORM/cruise/approach routing of independently trained frozen actors."""
import numpy as np
import torch
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack
from swarm_core.learning.routed_policy import RoutedPolicy
from swarm_core.learning.approach_router import LocalApproachRouter

class BatchedMissionPolicy:
 def __init__(self,translation,formation,alignment=None,acquisition_routing=None,parking=None,motion_gate=None):
  from swarm_core.goal_stop import GoalStopLatch
  from swarm_core.predictive_stop import PredictiveStopGate
  self.motion_gate=PredictiveStopGate(motion_gate) if motion_gate else None
  self.parking=[GoalStopLatch(parking) for _ in range(3)] if parking else []
  self.translation=RoutedPolicy(translation)
  self.history_formation=formation['contract'] in ('formation_history_v5','formation_history_risk_v6')
  if formation['contract'] not in ('formation_relational_v4','formation_history_v5','formation_history_risk_v6') or formation['local_dim']!=9 or formation['neighbor_dim']!=({'formation_history_v5':11,'formation_history_risk_v6':13}.get(formation['contract'],9)):raise ValueError('Unsupported formation actor contract.')
  args,self.formation,_=make_policy(3,local_dim=9,critic_dim=formation['critic_dim'],neighbor_dim=formation['neighbor_dim'],geometry_summary=True)
  self.formation.actor.load_state_dict(formation['actor']);self.formation.actor.eval()
  self.form_hidden,self.form_masks=hidden_inputs(1,args)
  self.rosbot_fresh_formation=None
  self.fresh_local_dim=9
  self.fresh_formation=None;self.formation_expert_routers=[]
  if formation.get('fresh_actor') is not None:
   from swarm_core.learning.formation_expert_router import LocalFormationExpertRouter
   fresh=formation['fresh_actor']
   if not self.history_formation or (fresh['contract'],fresh['local_dim']) not in (('formation_relational_v4',9),('formation_capabilities_v1',13),('formation_pose_v1',15)) or fresh['neighbor_dim']!=9:raise ValueError('Invalid fresh-message formation expert.')
   self.fresh_local_dim=fresh['local_dim']
   a,model,_=make_policy(3,local_dim=self.fresh_local_dim,neighbor_dim=9,critic_dim=fresh['critic_dim'],geometry_summary=True)
   model.actor.load_state_dict(fresh['actor']);model.actor.eval();self.fresh_formation=(model,*hidden_inputs(1,a))
   rosbot=formation.get('rosbot_fresh_actor')
   if rosbot is not None:
    for key in ('contract','local_dim','neighbor_dim','critic_dim','geometry_summary'):
     if rosbot[key]!=fresh[key]:raise ValueError(f'RosBot formation contract mismatch: {key}')
    a,model,_=make_policy(3,local_dim=self.fresh_local_dim,neighbor_dim=9,
                          critic_dim=rosbot['critic_dim'],geometry_summary=True)
    model.actor.load_state_dict(rosbot['actor']);model.actor.eval()
    self.rosbot_fresh_formation=(model,*hidden_inputs(1,a))
   self.formation_expert_routers=[LocalFormationExpertRouter(formation.get('fresh_actor_max_normalized_age')) for _ in range(3)]
  self.alignment_model=None;self.acquisition_routers=[]
  self.stationary_alignment=bool(alignment and alignment.get('stationary_alignment',False))
  if alignment is not None:
   from swarm_core.learning.acquisition_router import LocalAcquisitionRouter
   if alignment.get('component')!='alignment' or alignment['contract']!='mission_local_goal_v3' or alignment['local_dim']!=16 or alignment['neighbor_dim']!=9:raise ValueError('Unsupported learned alignment actor.')
   a,model,_=make_policy(3,local_dim=16,critic_dim=alignment['critic_dim'],neighbor_dim=9,geometry_summary=True,hold_std_scale=alignment.get('hold_std_scale'),local_only=alignment.get('local_only',False))
   model.actor.load_state_dict(alignment['actor']);model.actor.eval();self.alignment_model=(model,*hidden_inputs(1,a))
   from swarm_core.history_startup import HistoryAcquisitionRouter
   router_class=HistoryAcquisitionRouter if self.history_formation else LocalAcquisitionRouter
   self.acquisition_routers=[router_class(**(acquisition_routing or {})) for _ in range(3)]

  self.routers=[LocalApproachRouter(translation['horizon_m'],translation['epsilon_m'],translation.get('observation_horizon_m',.5)) for _ in range(3)]
 def reset(self):
  for router in [*self.routers,*self.acquisition_routers,*self.formation_expert_routers,*self.parking]:router.reset()
 def act(self,observations):
  raw=pack(observations)
  if raw.shape!=(3,36) or not np.isfinite(raw).all():raise ValueError('Expected three finite mission observations.')
  modes=[]
  for i,(router,row,o) in enumerate(zip(self.routers,raw,observations)):
   phase=row[9:12]
   if not np.isclose(phase.sum(),1.) or not np.all(np.isclose(phase,0)|np.isclose(phase,1)):raise ValueError('Invalid mission phase.')
   if row[9]>.5:mode='alignment' if o.get('startup_committed',False) and self.alignment_model else (self.acquisition_routers[i].select(o) if self.acquisition_routers else 'formation')
   else:mode=self.translation.select_model(row,router.select(row,goal_distance_m=o['goal_distance_m']))
   if mode=='formation' and self.formation_expert_routers:mode=self.formation_expert_routers[i].select(o)
   if mode=='formation_fresh' and i==1 and self.rosbot_fresh_formation:
    mode='formation_rosbot'
   modes.append(mode)
  actions=np.empty((3,2),dtype=np.float32)
  with torch.no_grad():
   for mode in ['formation',*(['formation_fresh'] if self.fresh_formation else []),*(['formation_rosbot'] if self.rosbot_fresh_formation else []),*(['alignment'] if self.alignment_model else []),*self.translation.models]:
    indices=[i for i,name in enumerate(modes) if name==mode]
    if not indices:continue
    if mode=='formation':
     model,h,m=self.formation,self.form_hidden,self.form_masks
     inputs=pack([dict(local_features=observations[i]['local_features'][:9],**observations[i]['formation_history']) for i in indices]) if self.history_formation else np.concatenate([raw[indices,:9],raw[indices,16:]],axis=1)
    elif mode in ('formation_fresh','formation_rosbot'):
     model,h,m=self.rosbot_fresh_formation if mode=='formation_rosbot' else self.fresh_formation
     inputs=self._fresh_inputs(raw[indices],[observations[i] for i in indices])
    elif mode=='alignment':model,h,m=self.alignment_model;inputs=raw[indices]
    else:model,h,m=self.translation.models[mode];inputs=raw[indices]
    output,_=model.act(inputs,np.repeat(h,len(indices),axis=0),np.repeat(m,len(indices),axis=0),deterministic=True)
    actions[indices]=output.numpy()
    if mode=='alignment' and self.stationary_alignment:actions[indices,0]=0.
  for i,stop in enumerate(self.parking):
   actions[i],parked=stop.apply(actions[i],observations[i])
   if parked:modes[i]="parked"
  if self.motion_gate:
   for i in range(3):
    actions[i],guarded=self.motion_gate.apply(actions[i],observations[i])
    if guarded:modes[i]="guarded_"+modes[i]
  return actions,modes

 def _fresh_inputs(self,raw,observations):
  parts=[raw[:,:9]]
  if self.fresh_local_dim==15:
   pose=np.asarray([o['formation_pose'] for o in observations],dtype=np.float32)
   if pose.shape!=(len(observations),2) or not np.isfinite(pose).all():
    raise ValueError('Expected two finite formation pose features per robot.')
   parts.append(pose)
  if self.fresh_local_dim>=13:
   parts.append(np.asarray([o['formation_capabilities'] for o in observations],dtype=np.float32))
  return np.concatenate([*parts,raw[:,16:]],axis=1)

 def act_local(self,observation,robot_id):
  """Infer exactly one robot action using only its supplied observation.

  robot_id indexes this instance's local routing state, never a remote state.
  A deployed instance must be owned by one robot and reset per mission.
  No critic, simulator object or other robots' observations are needed.
  """
  if type(robot_id) is not int or not 0<=robot_id<3:raise ValueError('Expected robot ID 0, 1 or 2.')
  raw=pack([observation])
  if raw.shape!=(1,36) or not np.isfinite(raw).all():raise ValueError('Expected one finite mission observation.')
  row=raw[0];phase=row[9:12]
  if not np.isclose(phase.sum(),1.) or not np.all(np.isclose(phase,0)|np.isclose(phase,1)):raise ValueError('Invalid mission phase.')
  if row[9]>.5:
   mode='alignment' if observation.get('startup_committed',False) and self.alignment_model else (self.acquisition_routers[robot_id].select(observation) if self.acquisition_routers else 'formation')
  else:
   mode=self.translation.select_model(row,self.routers[robot_id].select(row,goal_distance_m=observation['goal_distance_m']))
  if mode=='formation' and self.formation_expert_routers:mode=self.formation_expert_routers[robot_id].select(observation)
  if mode=='formation':
   model,h,m=self.formation,self.form_hidden,self.form_masks
   inputs=pack([dict(local_features=observation['local_features'][:9],**observation['formation_history'])]) if self.history_formation else np.concatenate([raw[:,:9],raw[:,16:]],axis=1)
  elif mode=='formation_fresh':
   if robot_id==1 and self.rosbot_fresh_formation:
    mode='formation_rosbot'
    model,h,m=self.rosbot_fresh_formation
   else:model,h,m=self.fresh_formation
   inputs=self._fresh_inputs(raw,[observation])
  elif mode=='alignment':model,h,m=self.alignment_model;inputs=raw
  else:model,h,m=self.translation.models[mode];inputs=raw
  with torch.no_grad():output,_=model.act(inputs,h,m,deterministic=True)
  action=output.numpy()[0].copy()
  if mode=='alignment' and self.stationary_alignment:action[0]=0.
  if self.parking:
   action,parked=self.parking[robot_id].apply(action,observation)
   if parked:mode="parked"
  if self.motion_gate:
   action,guarded=self.motion_gate.apply(action,observation)
   if guarded:mode="guarded_"+mode
  return action,mode
