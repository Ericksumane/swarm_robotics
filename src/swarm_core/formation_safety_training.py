"""Training-only near-clearance shaping with unchanged local predictive veto."""
import math
import numpy as np
from swarm_core.formation_env import FormationEnv
from swarm_core.predictive_stop import PredictiveStopGate
from swarm_core.geometry import formation_metrics

class FormationSafetyTrainingEnv(FormationEnv):
 def __init__(self,task,communication,reward_weights):
  self.safety=task['formation_safety_training'];keys=['target_clearance_m','near_weight','recovery_progress_weight']
  if not np.isfinite([self.safety[k] for k in keys]).all() or any(self.safety[k]<=0 for k in keys):raise ValueError('Positive finite formation safety settings required.')
  self.gate=PredictiveStopGate(self.safety['motion_gate']);super().__init__(task,communication,reward_weights)
 def clearance(self):return formation_metrics(self.simulation.states,self.task['team']['desired_pair_distance_m'],self.task['robot_defaults']['footprint_radius_m'])['minimum_clearance_m']
 def step(self,actions):
  if self.done:raise RuntimeError('Reset required after episode end.')
  observations=self.simulation.observe_all();executed=np.asarray(actions,dtype=np.float32).copy();previous=self.clearance()
  for i,o in enumerate(observations):
   # Only this private gate adapter needs a mission-shaped phase flag. The
   # learned formation actor retains its original local9/history11 contract.
   own=self.simulation.estimates[i];angle=own.heading;c,s=math.cos(angle),math.sin(angle);direction=np.asarray(self.task['mission']['direction_world']);direction=np.array([[c,s],[-s,c]])@direction
   local=np.concatenate([o['local_features'][:9],[1.,0.,0.],[0.,0.],direction]).astype(np.float32)
   if self.task['observation_contract']!='formation_history_v5':raise ValueError('This safety adaptation is history-formation only.')
   gate_observation=dict(local_features=local,formation_history=dict(neighbor_features=o['neighbor_features'],neighbor_mask=o['neighbor_mask']))
   executed[i],_=self.gate.apply(executed[i],gate_observation)
  obs,rewards,term,trunc,info=super().step(executed);current=self.clearance();target=self.safety['target_clearance_m'];before=float(np.clip((target-previous)/target,0.,1.));after=float(np.clip((target-current)/target,0.,1.))
  extra=dict(near_clearance=-self.simulation.dt*self.safety['near_weight']*after**2,near_recovery_progress=self.safety['recovery_progress_weight']*(before**2-after**2))
  info['reward_terms'].update(extra);info['minimum_clearance_m']=current;rewards+=np.float32(sum(extra.values()))
  return obs,rewards,term,trunc,info
