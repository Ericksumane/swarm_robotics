"""External physics state adapter: never integrates Python robot dynamics."""
import numpy as np
from swarm_core.simulation import Simulation
from swarm_core.dynamics import RobotState

class ExternalSimulation(Simulation):
 def __init__(self,task,communication):
  self.pending=None;self.origin_clock_s=None;super().__init__(task,communication)
 def initialize_physics(self,states,clock_s,seed):
  if not np.isfinite(clock_s) or clock_s<0:raise ValueError('Initial Gazebo clock must be finite and nonnegative.')
  self.origin_clock_s=float(clock_s);self.states=self.validate_states(states);self.step_index=0
  self.pending=None;self._measure();self.reset_network(seed)
 def validate_states(self,states):
  if len(states)!=self.task['team']['num_agents']:raise ValueError('External state count mismatch.')
  values=np.asarray([[s[k] for k in ['x','y','heading','speed','yaw_rate']] for s in states],dtype=float)
  if not np.isfinite(values).all():raise ValueError('External physics samples must be finite.')
  return [RobotState(*row) for row in values]
 def offer(self,states,clock_s):
  if self.origin_clock_s is None:raise RuntimeError('External physics initialization is required.')
  if self.pending is not None:raise RuntimeError('An external sample is already pending.')
  expected=self.origin_clock_s+(self.step_index+1)*self.dt
  if not np.isfinite(clock_s) or abs(clock_s-expected)>1e-6:raise ValueError(f'Gazebo control-clock mismatch: {clock_s} versus {expected}.')
  self.pending=self.validate_states(states)
 def advance(self,actions):
  if np.asarray(actions).shape!=(len(self.states),2) or not np.isfinite(actions).all():raise ValueError('Invalid command array.')
  if self.pending is None:raise RuntimeError('Fresh external physics feedback is required.')
  self.states=self.pending;self.pending=None;self.step_index+=1;self._measure()
  if self.step_index%self.transmission_steps==0:self._transmit()
  self.network.deliver(self.time_s)
