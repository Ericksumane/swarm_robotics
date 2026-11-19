"""JSON-line live inference: Gazebo supplies all post-startup physical states."""
from pathlib import Path
import argparse,hashlib,json,secrets,sys
import numpy as np
import torch
from swarm_core.mission_env import MissionEnv
from swarm_core.external_simulation import ExternalSimulation
from swarm_core.geometry import formation_metrics
from swarm_core.learning.staged_policy import BatchedMissionPolicy

def emit(value):print(json.dumps(value),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parking-entry-m',type=float);p.add_argument('--continuous-hold',action='store_true');p.add_argument('--seed',type=int,default=None);p.add_argument('--communication',default='good');p.add_argument('--time-limit',type=float,default=240.);p.add_argument('--bundle',default=None);p.add_argument('--linear-speed-scale',type=float,default=1.);p.add_argument('--formation-only',action='store_true');p.add_argument('--formation-speed',type=float,default=.80);p.add_argument('--pose-goal',action='store_true');p.add_argument('--start-aligned',action='store_true');a=p.parse_args()
 if a.seed is None:a.seed=secrets.randbits(32)
 if a.seed<0:p.error('Seed must be nonnegative.')
 if a.bundle is None:a.bundle=json.load(open('experiments/selected_startup_candidate.json'))['actor_bundle']
 torch.set_num_threads(1);bundle=torch.load(a.bundle,map_location='cpu',weights_only=True);task=bundle['task'];task['simulation']['episode_time_limit_s']=a.time_limit
 from swarm_core.mission_spec import completion_task
 task=completion_task(task,a.communication)
 task['simulation']['episode_time_limit_s']=a.time_limit
 from swarm_core.motion_profile import scaled_linear_speed
 task=scaled_linear_speed(task,a.linear_speed_scale)
 if not .80<=a.formation_speed<=1.:p.error('Formation speed must be between 0.80 and 1.00 m/s.')
 if a.pose_goal:task['formation_pose_goal']=True
 if a.formation_only:
  task['robot_defaults']['max_forward_speed_mps']=a.formation_speed
  for key in ('formation_peer_history','peer_history'):
   if key in task:task[key]['max_speed_mps']=a.formation_speed
 desired=task['team']['desired_pair_distance_m']
 radius=task['robot_defaults']['footprint_radius_m']
 comm=json.load(open(f'configs/communication/{a.communication}.json'));reward=json.load(open('configs/training/geometry_control.json'))
 env=MissionEnv(task,comm,reward);env.simulation=ExternalSimulation(task,comm);env.reset(a.seed)
 parking=bundle.get('parking')
 if a.parking_entry_m is not None:
  if parking is None or not 0<a.parking_entry_m<parking['entry_m']:
   p.error('Parking entry override must tighten the selected stop threshold.')
  parking={**parking,'entry_m':a.parking_entry_m,'exit_m':a.parking_entry_m+.03}
 policy=BatchedMissionPolicy(bundle['translation'],bundle['formation'],alignment=bundle['alignment'],acquisition_routing=bundle['acquisition_routing'],parking=parking,motion_gate=bundle.get('motion_gate'));policy.reset();last_actions=None;initialized=False;first_success_s=None;formation_hold_s=0.
 emit(dict(kind='ready',initialization_seed=a.seed,initial_states=[vars(s) for s in env.simulation.states],task=task,communication=comm,bundle_sha256=hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest(),transmitted_payload_size_bytes=comm['payload_size_bytes']+(8 if task.get('history_startup') else 0),parking=parking,physics_source='Gazebo external odometry; Python dynamics disabled'))
 for line in sys.stdin:
  request=json.loads(line)
  if request['op']=='initialize':
   if initialized:raise RuntimeError('Only one initialization is allowed per worker.')
   env.simulation.initialize_physics(request['states'],request['clock_s'],a.seed);env.initial_centroid=formation_metrics(env.simulation.states,desired,radius)['centroid_m'];initialized=True;info=dict(reason=None,time_s=0.,phases=env.phases.tolist())
   if a.start_aligned:
    if a.formation_only or formation_metrics(env.simulation.states,desired,radius)['max_pair_error_m']>task['mission']['formation_tolerance_m']:
     raise ValueError('Aligned cruise requires a valid starting triangle.')
    env.phases[:]=1
    env.starts=[np.array([s.x,s.y]) for s in env.simulation.estimates]
    env.goals=[start+task['mission']['translation_distance_m']*env.direction for start in env.starts]
    env.move_centroid=np.mean(env.starts,axis=0)
  elif request['op']=='step':
   if not initialized or last_actions is None:raise RuntimeError('Initialize Gazebo feedback first.')
   env.simulation.offer(request['states'],request['clock_s']);_,_,_,_,info=env.step(last_actions)
  else:raise ValueError('Unknown live operation.')
  if info.get('reason')=='success':
   if first_success_s is None:first_success_s=env.simulation.time_s
   if a.continuous_hold:
    env.done=False;env.success_steps=0;env.monitor.hold_steps=0
    env.task['simulation']['episode_time_limit_s']=1e12
  if first_success_s is not None and a.continuous_hold and not env.done:info['reason']='live_hold'
  if a.formation_only and request['op']=='step':
   geometry=formation_metrics(env.simulation.states,desired,radius)
   mission=task['mission']
   settled=geometry['max_pair_error_m']<=mission['formation_tolerance_m'] and all(
    abs(state.speed)<=mission['stop_linear_speed_tolerance_mps']
    and abs(state.yaw_rate)<=mission['stop_angular_speed_tolerance_radps']
    for state in env.simulation.states)
   if task.get('formation_pose_goal'):
    target=np.arctan2(env.direction[1],env.direction[0])
    settled=settled and all(abs(np.arctan2(np.sin(state.heading-target),np.cos(state.heading-target)))
                            <=mission['final_heading_tolerance_rad'] for state in env.simulation.states)
   formation_hold_s=formation_hold_s+env.simulation.dt if settled else 0.
   if not env.done and formation_hold_s>=mission['formation_ready_duration_s']-1e-9:
    env.done=True;info['reason']='formation_success'
   elif not env.done and any(env.phases!=0):
    env.done=True;info['reason']='unexpected_movement_phase'
  observations=env.observe_all()
  if env.done:actions=np.zeros((3,2));modes=['stopped']*3
  else:
   actions,modes=policy.act(observations)
   actions=env.limit_approach_speed(actions)
  last_actions=actions;limits=task['robot_defaults'];commands=[]
  for raw in actions:
   clipped=np.clip(raw,-1,1);scale=limits['max_forward_speed_mps'] if clipped[0]>=0 else limits['max_reverse_speed_mps'];commands.append([float(clipped[0]*scale),float(clipped[1]*limits['max_yaw_rate_radps'])])
  geometry=formation_metrics(env.simulation.states,desired,radius)
  emit(dict(kind='control',startup_proposals_s=env.simulation.start_proposals,startup_committed=env.simulation.start_committed,history_available=[int(o.get('formation_history',o)['neighbor_mask'].sum()) for o in observations],startup_ready_steps=env.ready_steps.tolist(),startup_ready_certificates=env.simulation.formation_ready,startup_heading_errors_rad=[float(abs(np.arctan2(np.sin(state.heading-np.arctan2(env.direction[1],env.direction[0])),np.cos(state.heading-np.arctan2(env.direction[1],env.direction[0]))))) for state in env.simulation.estimates],move_start_positions_m=[None if value is None else value.tolist() for value in env.starts],goals_m=[None if value is None else value.tolist() for value in env.goals],mission_completed=first_success_s is not None,first_success_s=first_success_s,commands=commands,actions=actions.tolist(),modes=modes,phases=env.phases.tolist(),time_s=env.simulation.time_s,done=env.done,reason=info.get('reason'),goal_errors_m=info.get('goal_errors_m'),max_pair_error_m=geometry['max_pair_error_m'],minimum_clearance_m=geometry['minimum_clearance_m'],valid_neighbors=[int(o['neighbor_mask'].sum()) for o in observations]))
  if env.done:return

if __name__=='__main__':
 try:main()
 except Exception as error:emit(dict(kind='error',error=str(error)));raise
