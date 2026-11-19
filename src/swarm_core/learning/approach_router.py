"""Local mission-state selection between two frozen learned actors."""
import math
import numpy as np

class LocalApproachRouter:
 def __init__(self,horizon_m=.5,epsilon_m=1e-5,observation_horizon_m=.5):
  if not math.isfinite(horizon_m) or not math.isfinite(epsilon_m) or not 0<epsilon_m<horizon_m:raise ValueError('Invalid approach horizon.')
  if not math.isfinite(observation_horizon_m) or observation_horizon_m<=0:raise ValueError('Invalid observation horizon.')
  self.horizon_m=horizon_m;self.epsilon_m=epsilon_m;self.observation_horizon_m=observation_horizon_m;self.reset()
 def reset(self):self.approach_latched=False
 def select(self,observation,goal_distance_m=None):
  raw=np.asarray(observation)
  if raw.ndim!=1 or len(raw)<16 or not np.isfinite(raw).all():raise ValueError('Expected one finite mission-local observation.')
  phase=raw[9:12]
  if not np.isclose(phase.sum(),1.) or not np.all(np.isclose(phase,0)|np.isclose(phase,1)):raise ValueError('Invalid phase encoding.')
  if phase[0]>.5:raise ValueError('This translation bundle does not implement formation acquisition.')
  observed_norm=float(np.linalg.norm(raw[12:14]))
  if goal_distance_m is None:
   if self.horizon_m>self.observation_horizon_m:raise ValueError('Unclipped own-goal distance is required for this routing horizon.')
   distance=observed_norm
  else:
   distance=float(goal_distance_m)
   if not math.isfinite(distance) or distance<0:raise ValueError('Own-goal distance must be finite and nonnegative.')
   if not np.isclose(observed_norm,min(distance,self.observation_horizon_m),atol=1e-4):raise ValueError('Own-goal distance is inconsistent with actor goal features.')
  self.approach_latched|=bool(distance<self.horizon_m-self.epsilon_m or phase[2]>.5)
  return 'approach' if self.approach_latched else 'cruise'
