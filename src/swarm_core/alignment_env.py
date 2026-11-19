"""MAPPO component task: preserve geometry, align bodies, and settle."""
import math
import numpy as np
from swarm_core.mission_env import MissionEnv,FORM
from swarm_core.geometry import formation_metrics

class AlignmentEnv(MissionEnv):
 def reset(self,seed=None):
  super().reset(seed)
  self.phases[:]=FORM;self.goals=[None]*len(self.phases);self.starts=[None]*len(self.phases);self.move_centroid=None
  return self.observe_all(),dict(time_s=0.,phases=self.phases.tolist())
 def _update_phases(self):pass
 def heading_errors(self):
  angle=math.atan2(self.direction[1],self.direction[0])
  return np.asarray([abs(math.atan2(math.sin(s.heading-angle),math.cos(s.heading-angle))) for s in self.simulation.states])
 def step(self,actions):
  if self.done:raise RuntimeError('Reset required after episode end.')
  if self.task.get('stationary_alignment',False):
   actions=np.array(actions,dtype=float,copy=True);actions[:,0]=0.
  before=self.heading_errors();self.simulation.advance(actions);errors=self.heading_errors();dt=self.simulation.dt;limits=self.task['robot_defaults'];mission=self.task['mission'];w=self.weights
  geometry=formation_metrics(self.simulation.states,self.task['team']['desired_pair_distance_m'],limits['footprint_radius_m'])
  terms=dict(heading_progress=w['alignment_progress']*float(np.mean(before-errors)),heading=-dt*w['alignment_heading']*float(np.mean(1-np.cos(errors))),linear=-dt*w['alignment_linear']*float(np.mean([(s.speed/limits['max_forward_speed_mps'])**2 for s in self.simulation.states])),turning=-dt*w['alignment_turning']*float(np.mean([(s.yaw_rate/limits['max_yaw_rate_radps'])**2 for s in self.simulation.states])),formation=-dt*w['formation']*geometry['formation_rmse_m']**2,time=-dt*w['time'])
  stopped=all(abs(s.speed)<=mission['stop_linear_speed_tolerance_mps'] and abs(s.yaw_rate)<=mission['stop_angular_speed_tolerance_radps'] for s in self.simulation.states)
  complete=max(errors)<=self.task['alignment']['heading_tolerance_rad'] and stopped and geometry['max_pair_error_m']<=mission['formation_tolerance_m']
  self.success_steps=self.success_steps+1 if complete else 0;self.monitor.hold_steps=self.success_steps
  area=self.task['operating_area'];r=limits['footprint_radius_m'];outside=any(not(area['x_bounds_m'][0]+r<=s.x<=area['x_bounds_m'][1]-r and area['y_bounds_m'][0]+r<=s.y<=area['y_bounds_m'][1]-r) for s in self.simulation.states)
  reason=None;term=False;trunc=False
  if geometry['collision']:reason='collision';term=True
  elif outside:reason='outside_area';term=True
  elif self.success_steps*dt>=mission['success_hold_duration_s']-1e-12:reason='success';term=True
  elif self.simulation.time_s>=self.task['simulation']['episode_time_limit_s']-1e-12:reason='time_limit';trunc=True
  terms['terminal']=w['success_bonus'] if reason=='success' else (-w['failure_penalty'] if reason in ('collision','outside_area') else 0.)
  self.done=term or trunc
  info=dict(time_s=self.simulation.time_s,reason=reason,terminated=term,truncated=trunc,reward_terms=terms,heading_errors_rad=errors.tolist(),formation_rmse_m=geometry['formation_rmse_m'],component='alignment_only_not_translation')
  return self.observe_all(),np.full(len(self.phases),sum(terms.values()),dtype=np.float32),term,trunc,info
