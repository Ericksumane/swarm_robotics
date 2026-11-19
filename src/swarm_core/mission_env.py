"""Leaderless form -> translate 5 m -> stop task, with local phase decisions.

Readiness is latched and broadcast via impaired neighbor messages. Each robot
starts after its own readiness and received readiness of every peer. There is
no hidden global phase signal. Under partitions, starts may be asynchronous;
this protocol is not a consensus or safety guarantee.
"""
from copy import deepcopy
from types import SimpleNamespace
import math
import numpy as np
from swarm_core.observations import append_formation_geometry
from swarm_core.simulation import Simulation
from swarm_core.geometry import formation_metrics
from swarm_core.goal_stop import limit_forward_for_arrival
from swarm_core.observations import build_local_features,build_neighbor_observation,formation_pose_features

FORM,MOVE,HOLD=0,1,2

class MissionEnv:
 def __init__(self,task,communication,reward_weights):
  self.task=deepcopy(task);self.communication=deepcopy(communication)
  self.reward_weights=deepcopy(reward_weights)
  mission=self.task['mission']
  self.startup_geometry_tolerance_m=mission.get('startup_geometry_tolerance_m',mission['formation_tolerance_m'])
  if not np.isfinite(self.startup_geometry_tolerance_m) or not 0<self.startup_geometry_tolerance_m<=mission['formation_tolerance_m']:raise ValueError('Startup geometry tolerance must be positive and no looser than formation tolerance.')
  self.startup_heading_tolerance_rad=mission.get('startup_heading_tolerance_rad')
  if self.startup_heading_tolerance_rad is not None and (not np.isfinite(self.startup_heading_tolerance_rad) or not 0<self.startup_heading_tolerance_rad<=math.pi):raise ValueError('Startup heading tolerance must be in (0,pi].')
  self.hold_entry_tolerance_m=mission.get('hold_entry_tolerance_m',mission['destination_tolerance_m'])
  self.hold_exit_tolerance_m=mission.get('hold_exit_tolerance_m',mission['destination_tolerance_m'])
  if not np.isfinite([self.hold_entry_tolerance_m,self.hold_exit_tolerance_m]).all() or not 0<self.hold_entry_tolerance_m<=self.hold_exit_tolerance_m:raise ValueError('Invalid HOLD entry/exit tolerances.')
  defaults=dict(formation=1.,formation_progress=10.,translation_progress=10.,proximity=10.,proximity_margin_m=.3,time=.01,settling=.05,success_bonus=20.,failure_penalty=50.)
  defaults.update(reward_weights);self.weights=defaults
  if any(not np.isfinite(v) or v<0 for k,v in defaults.items() if k!='reward_version'):raise ValueError('Invalid mission reward weights.')
  if defaults['proximity_margin_m']<=0:raise ValueError('Invalid proximity margin.')
  self.simulation=Simulation(task,communication)
  direction=np.asarray(task['mission']['direction_world'],dtype=float)
  if direction.shape!=(2,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:
   raise ValueError('Mission direction must be a finite nonzero 2D vector.')
  self.direction=direction/np.linalg.norm(direction)
  self.reset()

 def reset(self,seed=None):
  self.simulation.reset(seed);n=len(self.simulation.states)
  self.phases=np.zeros(n,dtype=int);self.ready_steps=np.zeros(n,dtype=int)
  self.goals=[None]*n;self.starts=[None]*n
  self.done=False;self.success_steps=0
  from swarm_core.history_startup import HistoryStartupCoordinator
  self.history_coordinators=[HistoryStartupCoordinator(self.task["history_startup"],self.simulation.dt,
   desired_distance_m=self.task["team"]["desired_pair_distance_m"]) for _ in range(n)] if self.task.get("history_startup") else []
  self.monitor=SimpleNamespace(hold_steps=0)
  self.initial_centroid=formation_metrics(self.simulation.states,self.task['team']['desired_pair_distance_m'],self.task['robot_defaults']['footprint_radius_m'])['centroid_m']
  self.move_centroid=None
  if self.task.get('start_in_formation', False):
   geometry=formation_metrics(self.simulation.states,self.task['team']['desired_pair_distance_m'],self.task['robot_defaults']['footprint_radius_m'])
   if geometry['max_pair_error_m']>self.task['mission']['formation_tolerance_m']:
    raise ValueError('Movement-stage reset must already satisfy formation tolerance.')
   self.phases[:]=MOVE
   self.starts=[np.array([s.x,s.y]) for s in self.simulation.estimates]
   self.goals=[start+self.task['mission']['translation_distance_m']*self.direction for start in self.starts]
   self.move_centroid=np.mean(self.starts,axis=0)
  if self.task.get('approach_reset_bank'):
   from swarm_core.approach_resets import sample_approach_reset
   actual_seed=self.task['seed'] if seed is None else seed
   states,starts,goals=sample_approach_reset(self.task['approach_reset_bank'],actual_seed,n,self.task['robot_defaults'],self.task['operating_area'])
   self.simulation.states=states
   self.simulation._measure()
   self.simulation.reset_network(actual_seed)
   self.starts=starts;self.goals=goals;self.phases[:]=MOVE
   self.initial_centroid=np.mean(starts,axis=0);self.move_centroid=self.initial_centroid.copy()
  return self.observe_all(),dict(time_s=0.,phases=self.phases.tolist())

 def _local_team(self,i):
  own=self.simulation.estimates[i];team=[own];messages=[]
  for j in range(len(self.phases)):
   if i==j:continue
   message,_,valid=self.simulation.network.read(i,j,self.simulation.time_s)
   if not valid:return None,None
   team.append(message);messages.append(message)
  return team,messages

 def _update_phases(self):
  mission=self.task['mission'];dt=self.simulation.dt
  required=math.ceil(mission['formation_ready_duration_s']/dt-1e-12)
  # Changes are based on delivered snapshots, never on other actors' phases.
  for i in range(len(self.phases)):
   own=self.simulation.estimates[i]
   if self.phases[i]==FORM and self.history_coordinators:
    history=self._history_observation(i)
    coordinator=self.history_coordinators[i]
    ready,proposal,start=coordinator.update(own,self.simulation.network.receivers[i],history,self.simulation.time_s,self.direction)
    self.simulation.formation_ready[i]=ready;self.simulation.start_proposals[i]=proposal
    self.simulation.start_committed[i]=coordinator.committed
    self.ready_steps[i]=round(coordinator.evidence.duration/dt)
    if start:
     self.phases[i]=MOVE;self.starts[i]=np.array([own.x,own.y]);self.goals[i]=self.starts[i]+mission['translation_distance_m']*self.direction
   elif self.phases[i]==FORM:
    team,messages=self._local_team(i)
    ready=False
    if team is not None:
     geometry=formation_metrics(team,self.task['team']['desired_pair_distance_m'],self.task['robot_defaults']['footprint_radius_m'])
     ready=geometry['max_pair_error_m']<=self.startup_geometry_tolerance_m and all(abs(s.speed)<=mission['stop_linear_speed_tolerance_mps'] and abs(s.yaw_rate)<=mission['stop_angular_speed_tolerance_radps'] for s in team)
    if ready and self.startup_heading_tolerance_rad is not None:
     direction_angle=math.atan2(self.direction[1],self.direction[0])
     ready=all(abs(math.atan2(math.sin(s.heading-direction_angle),math.cos(s.heading-direction_angle)))<=self.startup_heading_tolerance_rad for s in team)
    self.ready_steps[i]=self.ready_steps[i]+1 if ready else 0
    if self.ready_steps[i]>=required:self.simulation.formation_ready[i]=True
    if self.simulation.formation_ready[i] and messages is not None and all(m.formation_ready for m in messages):
     self.phases[i]=MOVE
     self.starts[i]=np.array([own.x,own.y])
     self.goals[i]=self.starts[i]+mission['translation_distance_m']*self.direction
   elif self.phases[i]==MOVE:
    if np.linalg.norm(np.array([own.x,own.y])-self.goals[i])<=self.hold_entry_tolerance_m:
     self.phases[i]=HOLD
   elif self.phases[i]==HOLD:
    # Resume correction if disturbance moves the robot away from its goal.
    if np.linalg.norm(np.array([own.x,own.y])-self.goals[i])>self.hold_exit_tolerance_m:
     self.phases[i]=MOVE

 def _history_observation(self,i):
  from swarm_core.communication.history import build_history_observation
  return build_history_observation(self.simulation.estimates[i],self.simulation.network.receivers[i],
   [j for j in range(len(self.phases)) if j!=i],self.simulation.time_s,self.task['team']['desired_pair_distance_m'],self.task['formation_peer_history'])

 def observe_all(self):
  result=[];dt=self.simulation.dt
  for i,own in enumerate(self.simulation.estimates):
   observation=build_neighbor_observation(own,self.simulation.network.receivers[i],[j for j in range(len(self.phases)) if j!=i],self.simulation.time_s,predict_to_now=self.task.get('observation_contract')=='mission_local_goal_time_aligned_v4')
   c,s=math.cos(own.heading),math.sin(own.heading)
   rotation=np.array([[c,s],[-s,c]])
   goal=np.zeros(2) if self.goals[i] is None else rotation@(self.goals[i]-[own.x,own.y])
   observation['mission_heading_error_rad']=float(abs(math.atan2(*(rotation@self.direction)[::-1])))
   observation['goal_distance_m']=None if self.goals[i] is None else float(np.linalg.norm(goal))
   if self.task.get('observation_contract') in ('mission_local_goal_v3','mission_local_goal_time_aligned_v4'):
    horizon=self.task.get('goal_observation_horizon_m')
    if horizon is None or not np.isfinite(horizon) or horizon<=0:
     raise ValueError('Local goal horizon must be positive and finite.')
    norm=float(np.linalg.norm(goal))
    goal=goal*min(1.,horizon/max(norm,1e-12))
   phase=np.eye(3)[self.phases[i]]
   local=build_local_features(own,self.task['robot_defaults'],self.task['team']['desired_pair_distance_m'])
   if self.task.get('observation_contract') in ('mission_relational_v2','mission_local_goal_v3','mission_local_goal_time_aligned_v4'):
    observation['neighbor_features']=append_formation_geometry(
        observation['neighbor_features'],observation['neighbor_mask'],self.task['team']['desired_pair_distance_m'])
   observation['local_features']=np.concatenate([local,phase,goal,rotation@self.direction]).astype(np.float32)
   if self.task.get('formation_peer_history'):observation['formation_history']=self._history_observation(i)
   if self.task.get('formation_pose_goal'):
    observation['formation_pose']=formation_pose_features(own.heading,self.direction)
   if self.task.get('formation_capabilities'):
    from swarm_core.mixed_commands import FORMATION_CAPABILITIES
    observation['formation_capabilities']=FORMATION_CAPABILITIES[i]
   observation['desired_pair_distance_m']=self.task['team']['desired_pair_distance_m']
   observation['startup_committed']=self.simulation.start_committed[i]
   observation['control_time_s']=self.simulation.time_s
   result.append(observation)
  return result

 def limit_approach_speed(self,actions):
  braking=self.task.get('braking_envelope_mps2')
  if braking is None:return actions
  limits=self.task['robot_defaults']
  return np.asarray([
   limit_forward_for_arrival(action,state,goal,limits,self.simulation.dt,braking)
   for action,state,goal in zip(actions,self.simulation.estimates,self.goals)
  ],dtype=np.float32)

 def step(self,actions):
  if self.done:raise RuntimeError('Reset required after episode end.')
  n=len(self.phases);mission=self.task['mission'];limits=self.task['robot_defaults'];dt=self.simulation.dt
  before=formation_metrics(self.simulation.states,self.task['team']['desired_pair_distance_m'],limits['footprint_radius_m'])
  moving_before=self.phases==MOVE
  direction_angle_before=math.atan2(self.direction[1],self.direction[0])
  angle_errors_before=np.array([abs(math.atan2(math.sin(s.heading-direction_angle_before),math.cos(s.heading-direction_angle_before))) for s in self.simulation.states])
  goal_angle_before=np.array([0. if self.goals[i] is None else abs(math.atan2(math.sin(math.atan2(self.goals[i][1]-s.y,self.goals[i][0]-s.x)-s.heading),math.cos(math.atan2(self.goals[i][1]-s.y,self.goals[i][0]-s.x)-s.heading))) for i,s in enumerate(self.simulation.states)])
  old_goal_errors=[None if self.goals[i] is None else float(np.linalg.norm(np.array([s.x,s.y])-self.goals[i])) for i,s in enumerate(self.simulation.states)]
  self.simulation.advance(self.limit_approach_speed(actions))
  self._update_phases()
  if self.move_centroid is None and np.all(self.phases>=MOVE):
   # Evaluation only; actors never receive this centroid.
   self.move_centroid=np.mean(np.array(self.starts),axis=0)
  geometry=formation_metrics(self.simulation.states,self.task['team']['desired_pair_distance_m'],limits['footprint_radius_m'])
  errors=[None if self.goals[i] is None else float(np.linalg.norm(np.array([s.x,s.y])-self.goals[i])) for i,s in enumerate(self.simulation.states)]
  movement_progress=sum(old-errors[i] for i,old in enumerate(old_goal_errors) if old is not None)/n
  clearance=geometry['pair_distances_m']-2*limits['footprint_radius_m']
  weights=self.weights;margin=weights['proximity_margin_m']
  terms=dict(formation=-dt*weights['formation']*geometry['formation_rmse_m']**2,
   formation_progress=weights["formation_progress"]*(before['formation_rmse_m']-geometry['formation_rmse_m']),
   translation_progress=weights["translation_progress"]*movement_progress,
   proximity=-dt*weights["proximity"]*float(np.mean(np.maximum(0,(margin-clearance)/margin)**2)),
   time=-dt*weights["time"],
   settling=-dt*weights["settling"]*float(np.mean([(s.speed/limits["max_forward_speed_mps"])**2+(s.yaw_rate/limits["max_yaw_rate_radps"])**2 for i,s in enumerate(self.simulation.states) if self.phases[i]!=MOVE] or [0.])))
  terms['formation_max_progress']=weights.get('formation_max_progress',0.)*(before['max_pair_error_m']-geometry['max_pair_error_m'])
  # Bounded soft constraint; true geometry is used for training rewards only.
  tolerance=mission['formation_tolerance_m']
  barrier_weight=weights.get('formation_barrier',0.)
  excess=max(0.,(geometry['max_pair_error_m']-.8*tolerance)/(.2*tolerance))
  terms['formation_barrier']=-dt*barrier_weight*min(excess**2,1.)

  # Optional mission shaping: reward behavior, never prescribe actions.
  moving=[s for i,s in enumerate(self.simulation.states) if self.phases[i]==MOVE]
  direction_angle=math.atan2(self.direction[1],self.direction[0])
  terms['heading']=-dt*weights.get('heading',0.)*float(np.mean([1.-math.cos(s.heading-direction_angle) for s in moving] or [0.]))
  absolute_angles=np.array([abs(math.atan2(math.sin(s.heading-direction_angle),math.cos(s.heading-direction_angle))) for s in self.simulation.states])
  terms['heading_absolute']=-dt*weights.get('heading_absolute',0.)*float(np.mean([absolute_angles[i] for i in range(n) if self.phases[i]==MOVE] or [0.]))
  terms['heading_progress']=weights.get('heading_progress',0.)*float(np.mean((angle_errors_before-absolute_angles)*moving_before))
  terms['turning_absolute']=-dt*weights.get('turning_absolute',0.)*float(np.mean([abs(s.yaw_rate)/limits['max_yaw_rate_radps'] for s in moving] or [0.]))
  goal_heading_errors=[1.-math.cos(math.atan2(self.goals[i][1]-s.y,self.goals[i][0]-s.x)-s.heading) for i,s in enumerate(self.simulation.states) if self.phases[i]==MOVE and self.goals[i] is not None]
  goal_angle_after=np.array([0. if self.goals[i] is None else abs(math.atan2(math.sin(math.atan2(self.goals[i][1]-s.y,self.goals[i][0]-s.x)-s.heading),math.cos(math.atan2(self.goals[i][1]-s.y,self.goals[i][0]-s.x)-s.heading))) for i,s in enumerate(self.simulation.states)])
  terms['goal_heading_absolute']=-dt*weights.get('goal_heading_absolute',0.)*float(np.mean([goal_angle_after[i] for i in range(n) if self.phases[i]==MOVE] or [0.]))
  terms['goal_heading_progress']=weights.get('goal_heading_progress',0.)*float(np.mean((goal_angle_before-goal_angle_after)*moving_before))
  terms['goal_heading']=-dt*weights.get('goal_heading',0.)*float(np.mean(goal_heading_errors or [0.]))
  terms['turning']=-dt*weights.get('turning',0.)*float(np.mean([(s.yaw_rate/limits['max_yaw_rate_radps'])**2 for s in moving] or [0.]))
  terms['speed_spread']=-dt*weights.get('speed_spread',0.)*float(np.var([s.speed/limits['max_forward_speed_mps'] for s in moving]) if moving else 0.)
  area=self.task['operating_area'];r=limits['footprint_radius_m']
  outside=any(not(area['x_bounds_m'][0]+r<=s.x<=area['x_bounds_m'][1]-r and area['y_bounds_m'][0]+r<=s.y<=area['y_bounds_m'][1]-r) for s in self.simulation.states)
  stopped=all(abs(s.speed)<=mission['stop_linear_speed_tolerance_mps'] and abs(s.yaw_rate)<=mission['stop_angular_speed_tolerance_radps'] for s in self.simulation.states)
  complete=all(e is not None and e<=mission['destination_tolerance_m'] for e in errors) and stopped and geometry['max_pair_error_m']<=mission['formation_tolerance_m']
  final_heading_tolerance=mission.get('final_heading_tolerance_rad')
  if final_heading_tolerance is not None:
   complete=complete and bool(np.all(absolute_angles<=final_heading_tolerance))
  self.success_steps=self.success_steps+1 if complete else 0
  self.monitor.hold_steps=self.success_steps
  reason=None;terminated=False;truncated=False
  if geometry['collision']:reason='collision';terminated=True
  elif outside:reason='outside_area';terminated=True
  elif self.success_steps*dt>=mission['success_hold_duration_s']-1e-12:reason='success';terminated=True
  elif self.simulation.time_s>=self.task['simulation']['episode_time_limit_s']-1e-12:reason='time_limit';truncated=True
  terms['terminal']=weights['success_bonus'] if reason=='success' else (-weights['failure_penalty'] if reason in ('collision','outside_area') else 0.)
  self.done=terminated or truncated
  displacement=None if self.move_centroid is None else float(np.dot(geometry['centroid_m']-self.move_centroid,self.direction))
  info=dict(time_s=self.simulation.time_s,phases=self.phases.tolist(),reason=reason,terminated=terminated,truncated=truncated,reward_terms=terms,goal_errors_m=errors,translation_m=displacement,formation_rmse_m=geometry['formation_rmse_m'])
  return self.observe_all(),np.full(n,sum(terms.values()),dtype=np.float32),terminated,truncated,info
