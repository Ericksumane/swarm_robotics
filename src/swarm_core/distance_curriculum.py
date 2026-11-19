"""Training-only mission-distance curriculum; never generates robot actions."""
from collections import deque
import math
import numpy as np

class DistanceCurriculum:
 def __init__(self,settings,seed):
  self.stages=settings['stages'];self.window=settings.get('window_episodes',8)
  self.threshold=settings.get('promotion_success_rate',.75);self.retention=settings.get('previous_stage_probability',.2)
  if not self.stages or type(self.window) is not int or self.window<1 or not 0<self.threshold<=1 or not 0<=self.retention<1:raise ValueError('Invalid distance curriculum')
  distances=[s['distance_m'] for s in self.stages]
  if any(not math.isfinite(s['distance_m']) or s['distance_m']<=0 or not math.isfinite(s['time_limit_s']) or s['time_limit_s']<=0 for s in self.stages) or any(b<=a for a,b in zip(distances,distances[1:])):raise ValueError('Invalid distance stages')
  self.stage=0;self.sampled_stage=0;self.history=deque(maxlen=self.window);self.rng=np.random.default_rng(seed)
 def apply(self,env):
  self.sampled_stage=self.stage
  if self.stage>0 and self.rng.random()<self.retention:self.sampled_stage=int(self.rng.integers(self.stage))
  selected=self.stages[self.sampled_stage]
  env.task['mission']['translation_distance_m']=selected['distance_m']
  env.task['simulation']['episode_time_limit_s']=selected['time_limit_s']
 def record(self,success):
  if self.sampled_stage!=self.stage:return
  self.history.append(bool(success))
  if len(self.history)==self.window and sum(self.history)/self.window>=self.threshold and self.stage+1<len(self.stages):
   self.stage+=1;self.history.clear()
 def status(self):
  return dict(stage=self.stage,sampled_stage=self.sampled_stage,distance_m=self.stages[self.stage]['distance_m'],recent_episodes=len(self.history),recent_success_rate=sum(self.history)/max(1,len(self.history)))
