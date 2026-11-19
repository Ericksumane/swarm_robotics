"""Local observed-geometry readiness selects learned acquisition/alignment."""
import math
import numpy as np

class LocalAcquisitionRouter:
 def __init__(self,desired_distance_m=1.,formation_tolerance_m=.1,max_linear_speed_mps=.03,ready_duration_s=2.,time_step_s=.1):
  values=[desired_distance_m,formation_tolerance_m,max_linear_speed_mps,ready_duration_s,time_step_s]
  if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Invalid local acquisition routing settings.')
  self.desired=desired_distance_m;self.tolerance=formation_tolerance_m;self.max_speed=max_linear_speed_mps;self.required=math.ceil(ready_duration_s/time_step_s-1e-12);self.reset()
 def reset(self):self.ready_steps=0;self.alignment_latched=False
 def select(self,observation):
  rows=np.asarray(observation['neighbor_features']);mask=np.asarray(observation['neighbor_mask']);local=np.asarray(observation['local_features'])
  if rows.shape!=(2,9) or mask.shape!=(2,) or local.shape!=(16,) or not np.isfinite(rows).all() or not np.isfinite(local).all():raise ValueError('Expected finite three-agent mission-local observations.')
  if not np.all(mask==1):self.reset();return 'formation'
  positions=np.vstack([np.zeros(2),rows[:,:2]])
  desired=observation.get('desired_pair_distance_m',self.desired)
  error=max(abs(np.linalg.norm(positions[i]-positions[j])-desired) for i in range(3) for j in range(i))
  if error>self.tolerance:self.reset();return 'formation'
  if self.alignment_latched:return 'alignment'
  slow=max(abs(local[0]),max(abs(rows[:,4])))<=self.max_speed
  self.ready_steps=self.ready_steps+1 if slow else 0
  self.alignment_latched=self.ready_steps>=self.required
  return 'alignment' if self.alignment_latched else 'formation'
