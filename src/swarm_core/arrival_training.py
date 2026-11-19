"""Training-only arrival shaping with the same local stop supervisors as deployment.

Actor inputs and evaluation criteria remain unchanged. Reset states contain no
expert actions. Already parked peers are represented faithfully rather than
silently trained as freely moving agents. Global truth is used for reward only.
"""
import copy
import numpy as np
from swarm_core.mission_env import MissionEnv,MOVE,HOLD
from swarm_core.goal_stop import GoalStopLatch
from swarm_core.predictive_stop import PredictiveStopGate

class ArrivalTrainingEnv(MissionEnv):
 def __init__(self,task,communication,reward_weights):
  settings=task['arrival_training']
  keys=['inner_target_m','distance_scale_m','excess_weight','excess_progress_weight','exit_event_penalty','deadlock_penalty','timeout_penalty']
  if not np.isfinite([settings[k] for k in keys]).all() or any(settings[k]<=0 for k in keys):raise ValueError('Arrival training settings must be positive and finite.')
  self.stop_settings=copy.deepcopy(settings['parking']);self.motion_gate=PredictiveStopGate(settings['motion_gate']) if settings.get('motion_gate') else None
  self.parking=[GoalStopLatch(self.stop_settings) for _ in range(task['team']['num_agents'])];self.last_actor_active_masks=np.ones((len(self.parking),1),dtype=np.float32)
  super().__init__(task,communication,reward_weights)
 def reset(self,seed=None):
  observations,info=super().reset(seed)
  for latch in self.parking:latch.reset()
  actual_seed=self.task['seed'] if seed is None else seed
  bank=self.task['approach_reset_bank'];rng=np.random.default_rng(np.random.SeedSequence([actual_seed,5181]));entry=bank['entries'][int(rng.integers(len(bank['entries'])))];parked=entry.get('parked',[False]*len(self.parking));phases=entry.get('phases',[MOVE]*len(self.parking))
  for i,latch in enumerate(self.parking):
   own=self.simulation.estimates[i];distance=float(np.linalg.norm(self.goals[i]-[own.x,own.y]));self.phases[i]=HOLD if phases[i]==HOLD and distance<=self.hold_exit_tolerance_m or distance<=self.hold_entry_tolerance_m else MOVE
   latch.parked=bool(parked[i] and distance<=self.stop_settings['exit_m'] and abs(own.speed)<=self.stop_settings['linear_tolerance_mps'] and abs(own.yaw_rate)<=self.stop_settings['yaw_tolerance_radps'])
  for latch, observation in zip(self.parking, self.observe_all()):
   tolerance=self.stop_settings.get('heading_tolerance_rad')
   if tolerance is not None and observation['mission_heading_error_rad']>tolerance:
    latch.parked=False
  self.last_actor_active_masks=np.asarray([[not p.parked] for p in self.parking],dtype=np.float32)
  return self.observe_all(),info
 def step(self,actions):
  if self.done:raise RuntimeError('Reset required after episode end.')
  observations=self.observe_all();executed=np.asarray(actions,dtype=np.float32).copy();before=np.asarray([o['goal_distance_m'] for o in observations]);active=[]
  for i,latch in enumerate(self.parking):executed[i],parked=latch.apply(executed[i],observations[i]);active.append(not parked)
  self.last_actor_active_masks=np.asarray(active,dtype=np.float32)[:,None]
  if self.motion_gate:
   for i in range(len(executed)):executed[i],_=self.motion_gate.apply(executed[i],observations[i])
  obs,rewards,term,trunc,info=super().step(executed);after=np.asarray(info['goal_errors_m']);active=np.asarray(active,dtype=bool);w=self.task['arrival_training']
  def excess(errors):
   values=np.clip(np.maximum(errors-w['inner_target_m'],0.)/w['distance_scale_m'],0.,1.)
   return float(values[active].max()) if active.any() else 0.
  terms=dict(arrival_excess=-self.simulation.dt*w['excess_weight']*excess(after),arrival_excess_progress=w['excess_progress_weight']*(excess(before)-excess(after)),arrival_exit_event=-w['exit_event_penalty']*float(np.mean((before<=self.stop_settings['entry_m']) & (after>self.stop_settings['exit_m']) & active)))
  if all(p.parked for p in self.parking) and info['reason'] is None and self.success_steps==0:
   # In the unchanged deployment latch this cannot recover without disturbance.
   # End this TRAINING failure early instead of wasting the remaining episode.
   terms['arrival_park_deadlock']=-w['deadlock_penalty'];term=True;self.done=True;info.update(reason='training_park_deadlock',terminated=True,truncated=False)
  heading_weight=w.get('final_heading_weight',0.)
  if heading_weight:
   heading=np.asarray([o['mission_heading_error_rad'] for o in obs])
   terms['arrival_heading']=-self.simulation.dt*heading_weight*float(np.mean(1.-np.cos(heading)))
  if trunc:terms['arrival_timeout']=-w['timeout_penalty']
  info['reward_terms'].update(terms);rewards=rewards+np.float32(sum(terms.values()))
  return obs,rewards,term,trunc,info
