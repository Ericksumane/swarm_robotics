"""Live Gazebo physics -> odometry -> frozen MAPPO -> velocity commands."""
from pathlib import Path
import argparse,datetime,json,math,os,secrets,select,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swarm_core.gazebo_world import build_live_world
from gz.transport13 import Node
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.world_control_pb2 import WorldControl

class PhysicsTransport:
 def __init__(self,control_process,step_s=.005):
  self.step_s=step_s;self.control_process=control_process;self.node=Node();self.condition=threading.Condition();self.clock=None;self.odometry={};self.callbacks=[]
  def clock(msg):
   with self.condition:self.clock=msg.sim.sec+msg.sim.nsec*1e-9;self.condition.notify_all()
  self.callbacks.append(clock);self.node.subscribe(Clock,'/clock',clock)
  for i in range(3):
   def odom(msg,index=i):
    q=msg.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    value=dict(x=msg.pose.position.x,y=msg.pose.position.y,heading=yaw,speed=msg.twist.linear.x,yaw_rate=msg.twist.angular.z,z=msg.pose.position.z,up_z=1-2*(q.x*q.x+q.y*q.y))
    stamp=msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
    with self.condition:self.odometry[index]=(stamp,value);self.condition.notify_all()
   self.callbacks.append(odom);self.node.subscribe(Odometry,f'/model/robot_{i}/odometry',odom)
  self.publishers=[self.node.advertise(f'/model/robot_{i}/cmd_vel',Twist) for i in range(3)]
 def control(self,steps=0,timeout=1000):
  self.control_process.stdin.write(json.dumps(dict(steps=steps,timeout=timeout))+chr(10));self.control_process.stdin.flush()
  if not select.select([self.control_process.stdout],[],[],timeout/1000.+2.)[0]:return False
  reply=json.loads(self.control_process.stdout.readline())
  if not reply['accepted'] and steps:print('Gazebo control response:',reply,flush=True)
  return reply['accepted']
 def commands(self,values):
  for publisher,(v,w) in zip(self.publishers,values):
   message=Twist();message.linear.x=float(v);message.angular.z=float(w);publisher.publish(message)
 def advance(self,steps,first=False):
  before=self.clock or 0.;target=before+(steps-(1 if first else 0))*self.step_s
  if not self.control(steps):raise RuntimeError('Gazebo rejected physics stepping.')
  with self.condition:
   if not self.condition.wait_for(lambda:self.clock is not None and self.clock>=target-1e-8,timeout=10.):raise RuntimeError(f'Gazebo clock timeout at {self.clock}, expected {target}.')
  if first:time.sleep(.03)
  with self.condition:
   clock=self.clock
   if not self.condition.wait_for(lambda:len(self.odometry)==3 and all(s>=clock-(.01001 if self.step_s==.001 else .00501) for s,_ in self.odometry.values()),timeout=5.):raise RuntimeError('Fresh Gazebo odometry from all three robots is unavailable.')
   samples=[dict(self.odometry[i][1]) for i in range(3)];ages=[clock-self.odometry[i][0] for i in range(3)]
  return clock,samples,ages

class Go2StrideControl:
 def __init__(self):
  import rclpy
  from rcl_interfaces.srv import SetParameters
  self.context=rclpy.Context()
  rclpy.init(context=self.context)
  from rclpy.executors import SingleThreadedExecutor
  self.executor=SingleThreadedExecutor(context=self.context)
  self.node=rclpy.create_node('swarm_stride_control',context=self.context)
  self.executor.add_node(self.node)
  self.client=self.node.create_client(SetParameters,'/swarm_go2/robot_velocity_handler/set_parameters')

 def set(self, scale):
  from rcl_interfaces.srv import SetParameters
  from rclpy.parameter import Parameter
  if not self.client.wait_for_service(timeout_sec=8.):
   raise RuntimeError('Go2 stride parameter service is unavailable.')
  request=SetParameters.Request(parameters=[Parameter('forward_stride_scale',value=scale).to_parameter_msg()])
  future=self.client.call_async(request)
  self.executor.spin_until_future_complete(future,timeout_sec=8.)
  response=future.result()
  if response is None or not response.results or not response.results[0].successful:
   reason=response.results[0].reason if response is not None and response.results else 'no response'
   raise RuntimeError(f'Go2 stride update failed: {reason}')

 def close(self):
  self.executor.shutdown()
  self.node.destroy_node()
  self.context.try_shutdown()


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mixed',action='store_true');parser.add_argument('--mixed-parking-entry',type=float);parser.add_argument('--startup-bundle');parser.add_argument('--linear-speed-scale',type=float,default=1.,help='Experimental forward/reverse speed multiplier, 1.0 to 1.15.');parser.add_argument('--keep-controlling',action='store_true');parser.add_argument('--end-on-success',action='store_true');parser.add_argument('--seed',type=int,default=None,help='Fixed reproducible seed; omitted selects a new random seed.');parser.add_argument('--communication',choices=['nominal','good','moderate','severe'],default='good');parser.add_argument('--time-limit',type=float,default=240.);parser.add_argument('--duration',type=float);parser.add_argument('--headless',action='store_true');parser.add_argument('--fast',action='store_true');parser.add_argument('--real-time-factor',type=float,default=.99,help='Wall-clock pacing target; --fast disables pacing.');parser.add_argument('--probe',action='store_true');parser.add_argument('--no-keep-open',action='store_true');parser.add_argument('--output');parser.add_argument('--formation-only',action='store_true');parser.add_argument('--formation-speed',type=float,default=.80);parser.add_argument('--mixed-actuator-probe',action='store_true');parser.add_argument('--hard-formation-start',action='store_true');parser.add_argument('--rosbot-reference',type=Path);parser.add_argument('--reference-feedback',action='store_true');parser.add_argument('--reference-replan',action='store_true');parser.add_argument('--random-starts',action='store_true');parser.add_argument('--cruise-actuator-probe',action='store_true');parser.add_argument('--cooperative-reference',type=Path);parser.add_argument('--cooperative-feedback',action='store_true');parser.add_argument('--cooperative-translation',action='store_true');parser.add_argument('--cooperative-cruise-speed',type=float,default=.34);parser.add_argument('--cooperative-adaptive-speed',action='store_true');parser.add_argument('--go2-speed-sweep',action='store_true');parser.add_argument('--cooperative-maneuver-scale',type=float,default=1.);parser.add_argument('--go2-forward-stride-scale',type=float,default=1.);parser.add_argument('--go2-stride-cruise-only',action='store_true');parser.add_argument('--wheeltec-arc-actor',type=Path);parser.add_argument('--anchor-pose-actors',type=Path);parser.add_argument('--translation-speed-actor',type=Path);parser.add_argument('--wheeltec-reversal-recovery',action='store_true');parser.add_argument('--wheeltec-hold-spacing',action='store_true');parser.add_argument('--cooperative-terminal-recovery',action='store_true');args=parser.parse_args()
 if args.cooperative_terminal_recovery and not args.cooperative_translation:
  parser.error('--cooperative-terminal-recovery requires cooperative translation.')
 if args.wheeltec_hold_spacing and (not args.cooperative_feedback or args.cooperative_translation or not args.wheeltec_reversal_recovery):
  parser.error('--wheeltec-hold-spacing requires cooperative pose feedback and reversal recovery.')
 if args.translation_speed_actor and not args.cooperative_translation:
  parser.error('The translation speed actor requires cooperative translation.')
 if args.anchor_pose_actors and not args.cooperative_feedback:
  parser.error('Anchor pose actors require cooperative feedback.')
 if (args.wheeltec_arc_actor or args.wheeltec_reversal_recovery) and not args.cooperative_feedback:
  parser.error('WheelTec arc execution and reversal recovery require cooperative feedback.')
 if not math.isfinite(args.formation_speed) or not .80<=args.formation_speed<=1.:parser.error('Formation speed must be between 0.80 and 1.00 m/s.')
 if args.random_starts and (not args.mixed or args.probe or args.rosbot_reference or args.mixed_actuator_probe):
  parser.error("Random starts require a mixed policy run.")
 if args.hard_formation_start and not (args.mixed and args.formation_only):
  parser.error('Hard formation starts require --mixed and --formation-only.')
 if args.reference_replan:
  args.reference_feedback=True
 if args.reference_feedback and not args.rosbot_reference:
  parser.error("Reference feedback requires --rosbot-reference.")
 if args.cooperative_feedback and not (args.cooperative_reference or args.random_starts):
  parser.error('Cooperative feedback requires --cooperative-reference or --random-starts.')
 if args.cooperative_translation and not args.cooperative_feedback:
  parser.error('Cooperative translation requires --cooperative-feedback.')
 if args.cooperative_adaptive_speed and not args.cooperative_translation:
  parser.error('Adaptive speed requires --cooperative-translation.')
 if not math.isfinite(args.cooperative_cruise_speed) or not 0.<args.cooperative_cruise_speed<=1.60:
  parser.error('Cooperative cruise speed must be in (0, 1.60] m/s.')
 if not math.isfinite(args.cooperative_maneuver_scale) or not 1.<=args.cooperative_maneuver_scale<=1.25:
  parser.error('Cooperative maneuver scale must be in [1.0, 1.25].')
 if args.cooperative_maneuver_scale!=1. and not args.cooperative_feedback:
  parser.error('Cooperative maneuver scale requires --cooperative-feedback.')
 if not math.isfinite(args.go2_forward_stride_scale) or not 1.<=args.go2_forward_stride_scale<=1.25:
  parser.error('Go2 forward stride scale must be in [1.0, 1.25].')
 if args.go2_forward_stride_scale!=1. and not args.mixed:
  parser.error('Go2 forward stride scale requires --mixed.')
 if args.go2_stride_cruise_only and not (args.cooperative_translation and args.go2_forward_stride_scale>1.):
  parser.error('Cruise-only stride requires cooperative translation and a stride scale above 1.0.')
 cooperative=None
 if args.cooperative_reference:
  if not args.mixed or args.formation_only or args.random_starts or args.rosbot_reference or args.mixed_actuator_probe or args.cruise_actuator_probe or args.duration is not None:
   parser.error('Cooperative reference requires --mixed and its recorded starts and duration.')
  records=(json.loads(line) for line in args.cooperative_reference.read_text().splitlines())
  cooperative=next((record for record in records if record['seed']==args.seed),None)
  if cooperative is None:parser.error('No cooperative demonstration for the requested seed.')
  args.probe=True
  args.duration=(130. if args.cooperative_translation else 90.) if args.cooperative_feedback else len(cooperative['samples'])*.1
 reference=None
 if args.rosbot_reference:
  if not args.mixed or args.formation_only or args.mixed_actuator_probe or args.duration is not None:
   parser.error('The RosBot reference requires --mixed and its own probe duration.')
  records=[json.loads(line) for line in args.rosbot_reference.read_text().splitlines()]
  reference=next((record for record in records if record['seed']==args.seed and record['found']),None)
  if reference is None:parser.error('No reference path for the requested seed.')
  # 0.18 m primitives at 0.15 m/s and a 0.1 s command period.
  args.probe=True
  args.duration=90. if args.reference_replan else len(reference['primitives'])*(4. if args.reference_feedback else 1.2)+3.
 if args.go2_speed_sweep and not args.cruise_actuator_probe:
  parser.error('Go2 speed sweep requires --cruise-actuator-probe.')
 if args.cruise_actuator_probe:
  if not args.mixed or args.formation_only or args.rosbot_reference or args.mixed_actuator_probe or args.duration is not None:
   parser.error('The cruise probe requires --mixed and uses a fixed duration.')
  args.probe=True
  args.duration=45. if args.go2_speed_sweep else 14.
 if args.mixed_actuator_probe:
  if not args.mixed or args.duration is not None:
   parser.error('The mixed actuator probe requires --mixed and uses a fixed duration.')
  args.probe=True
  args.duration=11.
 if args.formation_only and args.probe:parser.error('Formation evaluation requires a policy.')
 if not math.isfinite(args.time_limit) or args.time_limit<=0 or (args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0)):parser.error('Time limits must be positive and finite.')
 if not math.isfinite(args.real_time_factor) or not 0<args.real_time_factor<=1:parser.error('Real-time factor must be in (0, 1].')
 if not math.isfinite(args.linear_speed_scale) or not 1.<=args.linear_speed_scale<=1.15:parser.error('Linear speed scale must be within [1.0, 1.15].')
 seed_mode='random' if args.seed is None else 'fixed'
 if args.seed is None:args.seed=secrets.randbits(32)
 if args.seed<0:parser.error('Seed must be nonnegative.')
 if args.random_starts and args.cooperative_feedback:
  if args.cooperative_reference or not args.cooperative_translation:
   parser.error('Random cooperative starts require translation and no recorded reference.')
  from check_rosbot_formation_path import check
  from swarm_core.initialization import sample_initial_states
  task=json.loads((ROOT/'configs/tasks/mixed_formation_nominal080.json').read_text())
  poses=[(s.x,s.y,s.heading) for s in sample_initial_states(task,args.seed)]
  plan=check(args.seed,poses,task,choose_reachable=True)
  if not plan['found']:
   parser.error(f'No WheelTec route found for initialization seed {args.seed}.')
  goals=[(*plan['staged_anchors'][0],0.),(*plan['goal'],0.),(*plan['staged_anchors'][1],0.)]
  cooperative=dict(samples=[dict(robots=[dict(pose=pose,target_world=goal)
                                       for pose,goal in zip(poses,goals)])])
  args.probe=True
  args.duration=130.
 print(f'Initialization seed: {args.seed} ({seed_mode}); recorded in manifest.json',flush=True)
 os.chdir(ROOT);
 if args.mixed:os.environ.setdefault('ROS_DOMAIN_ID','43')
 os.environ['GZ_PARTITION']='swarm_live_'+str(os.getpid());os.environ.setdefault('LP_NUM_THREADS','2')
 out=Path(args.output) if args.output else ROOT/'experiments/runs'/('gazebo_live_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
 out.mkdir(parents=True,exist_ok=False);children=[];handles=[];transport=None;worker=None;stride_control=None
 def reply(timeout=30.):
  if not select.select([worker.stdout],[],[],timeout)[0]:raise RuntimeError('Live policy worker response timed out.')
  line=worker.stdout.readline()
  if not line:raise RuntimeError('Live policy worker closed; see policy.log.')
  value=json.loads(line)
  if value.get('kind')=='error':raise RuntimeError(value['error'])
  return value
 def rpc(request):worker.stdin.write(json.dumps(request)+chr(10));worker.stdin.flush();return reply(10.)
 def start(name,command,env=None):
  handle=(out/(name+'.log')).open('w');handles.append(handle);process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env);children.append(process);return process
 try:
  if args.probe:
   hello=dict(initial_states=[dict(x=0.,y=y,heading=0.,speed=0.,yaw_rate=0.) for y in [0.,2.,-2.]],physics_source='Gazebo actuator probe; no learned-policy result')
  else:
   policy_env=dict(os.environ);policy_env.pop('PYTHONPATH',None);policy_env.pop('PYTHONHOME',None)
   policy_python=ROOT/'.venv/bin/python'
   if args.mixed:
    policy_python=Path(os.environ.get('SWARM_POLICY_PYTHON',str(ROOT/'.venv/bin/python')))
    policy_env['PYTHONPATH']=str(ROOT/'src')
   policy_command=[str(policy_python),str(ROOT/'scripts/live_policy_worker.py'),'--seed',str(args.seed),'--communication',args.communication,'--time-limit',str(args.time_limit),'--linear-speed-scale',str(args.linear_speed_scale)]
   if args.mixed and args.mixed_parking_entry is not None:policy_command.extend(['--parking-entry-m',str(args.mixed_parking_entry)])
   if args.formation_only:policy_command.extend(['--formation-only','--formation-speed',str(args.formation_speed)])
   if args.startup_bundle:policy_command.extend(['--bundle',args.startup_bundle])
   if not args.formation_only and (not args.headless or args.keep_controlling) and not args.end_on_success:policy_command.append('--continuous-hold')
   error=(out/'policy.log').open('w');handles.append(error);worker=subprocess.Popen(policy_command,cwd=ROOT,env=policy_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1);children.append(worker);hello=reply()
  world=out/'world.sdf'
  limits=hello.get('task',{}).get('robot_defaults')
  if args.mixed:
   from swarm_core.mixed_gazebo_world import build_mixed_world
   side=hello.get('task',{}).get('team',{}).get('desired_pair_distance_m',1.)
   positions=[(side/math.sqrt(3),0.),(-side/(2*math.sqrt(3)),side/2),(-side/(2*math.sqrt(3)),-side/2)]
   starts=[dict(x=x,y=y,heading=0.,speed=0.,yaw_rate=0.) for x,y in positions]
   if args.formation_only or (args.random_starts and not cooperative):
    from swarm_core.initialization import sample_initial_states
    task=hello['task']
    sampling={**task,'initialization':dict(
     position_x_range_m=[-1.5,1.5],position_y_range_m=[-1.5,1.5],
     heading_range_rad=[-math.pi,math.pi],min_center_distance_m=1.05,
     initial_linear_speed_mps=0.,initial_angular_speed_radps=0.,max_sampling_attempts=1000)}
    if args.hard_formation_start:
     sampling['initialization']['hard_dispersed']={
      'probability':1.,'min_center_distance_m':1.65}
    starts=[vars(state) for state in sample_initial_states(sampling,args.seed)]
   if args.mixed_actuator_probe or args.cruise_actuator_probe:
    starts=hello["initial_states"]
    for state,y in zip(starts,(0.,4.,-4.)):state["y"]=y
   if reference:
    poses=[(*reference['staged_anchors'][0],0.),reference['start'],
           (*reference['staged_anchors'][1],0.)]
    starts=[dict(x=x,y=y,heading=heading,speed=0.,yaw_rate=0.)
            for x,y,heading in poses]
   if cooperative:
    starts=[dict(x=x,y=y,heading=heading,speed=0.,yaw_rate=0.)
            for x,y,heading in (robot['pose'] for robot in cooperative['samples'][0]['robots'])]
   world.write_text(build_mixed_world(out.resolve(),starts,limits))
   start('robot_description',['ros2','run','robot_state_publisher','robot_state_publisher',str(out.resolve()/'go2.urdf'),'--ros-args','-r','__ns:=/swarm_go2'])
   start('gait',['ros2','run','quadropted_controller','robot_controller_gazebo.py','--ros-args','-r','__ns:=/swarm_go2','-p','use_sim_time:=true'])
   start('velocity_adapter',['ros2','run','quadropted_controller','cmd_vel_pub.py','--ros-args','-r','__ns:=/swarm_go2','-p',f'forward_stride_scale:={1. if args.go2_stride_cruise_only else args.go2_forward_stride_scale}'])
   start('bridge',['ros2','run','ros_gz_bridge','parameter_bridge','/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock','/model/robot_0/cmd_vel@geometry_msgs/msg/Twist[gz.msgs.Twist','--ros-args','-r','/model/robot_0/cmd_vel:=/swarm_go2/cmd_vel'])
  else:
   world.write_text(build_live_world(hello['initial_states'],real_time_factor=0.,limits=limits))
  server=start('server',['gz','sim','-s','-v','2',str(world)])
  gui=None
  if not args.headless:gui=start('gui',['gz','sim','-g','--gui-config',str(ROOT/'gazebo/live_gui.config')])
  control_error=(out/'clock_service.log').open('w');handles.append(control_error)
  control_process=subprocess.Popen([sys.executable,str(ROOT/'scripts/gazebo_clock_service.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=control_error,text=True,bufsize=1);children.append(control_process)
  transport=PhysicsTransport(control_process,.001 if args.mixed else .005);ready=False
  for _ in range(100):
   if server.poll() is not None:raise RuntimeError('Gazebo server exited; see server.log.')
   if transport.control(timeout=200) and all(p.has_connections() for p in transport.publishers):ready=True;break
   time.sleep(.1)
  if not ready:raise RuntimeError('Gazebo physics/control service did not become ready.')
  transport.commands([[0.,0.]]*3)
  if args.mixed:
   spawner=start('spawner',['ros2','run','controller_manager','spawner','joint_group_controller','joint_state_broadcaster','-c','/swarm_go2/controller_manager','--controller-manager-timeout','20'])
   clock,states,ages=transport.advance(25,first=True)
   deadline=time.monotonic()+30
   while spawner.poll() is None and time.monotonic()<deadline:
    clock,states,ages=transport.advance(25)
    time.sleep(.01)
   if spawner.poll()!=0:raise RuntimeError('Go2 controller setup failed; see spawner.log.')
   walking=start('walk_mode',['ros2','service','call','/swarm_go2/robot_behavior_command','quadropted_msgs/srv/RobotBehaviorCommand',"{command: 'walk'}"])
   for _ in range(100):
    clock,states,ages=transport.advance(25)
    time.sleep(.02)
   if walking.poll()!=0 or 'success=True' not in (out/'walk_mode.log').read_text():raise RuntimeError('Go2 gait did not become active.')
   for _ in range(60):
    clock,states,ages=transport.advance(25)
    time.sleep(.025)
  else:
   clock,states,ages=transport.advance(100,first=True)
  origin=clock
  if args.mixed and (states[0]['z']<.18 or states[0]['up_z']<.8):
   raise RuntimeError(f"Go2 fell before policy activation: {states[0]}")
  response=rpc(dict(op='initialize',states=states,clock_s=clock)) if worker else dict(commands=[[.1,0.],[0.,0.],[0.,0.]],modes=['probe']*3,time_s=0.,done=False,reason=None)
  latest=ROOT/'experiments/runs/gazebo_live';latest.mkdir(exist_ok=True)
  metadata=dict(platforms=['Go2 EDU (gait model)','RosBot Pro (provisional Ackermann)','Jackal J100 (standard)'] if args.mixed else None,initialization_seed_mode=seed_mode,pid=os.getpid(),partition=os.environ['GZ_PARTITION'],output=str(out),settings={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},hello=hello,physics='Gazebo rigid-body and wheel contacts',feedback='Ideal Gazebo world-frame OdometryPublisher; not wheel localization',initial_physical_states=states)
  telemetry=None
  if args.cooperative_feedback and args.communication!='nominal':
   from cooperative_telemetry import CooperativeTelemetry
   profile=json.loads((ROOT/'configs/communication'/f'{args.communication}.json').read_text())
   telemetry=CooperativeTelemetry(profile,args.seed,len(states))
   metadata['cooperative_telemetry']=profile
  if cooperative:
   metadata['cooperative_initial_goals']=[robot['target_world'] for robot in cooperative['samples'][0]['robots']]
  if args.random_starts and cooperative:
   metadata['cooperative_preflight_plan']=plan
  (out/'manifest.json').write_text(json.dumps(metadata,indent=2));(latest/'latest.json').write_text(json.dumps(metadata,indent=2))
  if args.wheeltec_arc_actor:
   label='LIVE HYBRID: learned WheelTec arc execution with programmed planning and cooperative control.'
  elif args.cooperative_feedback:
   label='LIVE CLOSED LOOP: Gazebo odometry -> cooperative controller -> cmd_vel.'
  elif args.probe:
   label='PHYSICS PROBE: scripted actuator diagnostic.'
  else:
   label='LIVE CLOSED LOOP: Gazebo physics -> odometry -> MAPPO -> cmd_vel.'
  print(label,flush=True);print('Output:',out,flush=True)
  started=time.monotonic();tick=0;last_report=-1;initial=states;done_reason=None;success_reported=False
  translation=None;formation_completed_s=None
  terminal_controller=None;terminal_started_s=None
  terminal_correction_ticks=0;terminal_brake_ticks=0
  stride_scale=1. if args.go2_stride_cruise_only else args.go2_forward_stride_scale
  previous_go2_forward=0.
  stride_changes=[]
  precision_started_s=None
  precision_settling=False
  terminal_pair_tolerance_m=.05
  wheeltec_actor_calls=0
  anchor_actor_calls={0:0,2:0}
  translation_actor_calls=0
  if args.translation_speed_actor:
   import numpy as np
   artifact=json.loads(args.translation_speed_actor.read_text())
   if artifact['contract']!='translation_speed_v1':
    raise ValueError('Incompatible translation speed actor.')
   speed_weights={key:np.asarray(value,dtype=np.float32) for key,value in artifact['weights'].items()}
   def translation_actor(features,_reference_speed):
    nonlocal translation_actor_calls
    translation_actor_calls+=1
    hidden=np.tanh(speed_weights['0.weight']@np.asarray(features,dtype=np.float32)+speed_weights['0.bias'])
    return float((speed_weights['2.weight']@hidden+speed_weights['2.bias'])[0])
  local_clearance_stops=0
  clearance_blocked_ticks=0
  formation_clearance_replans=0
  synchronize_formation_start=bool(args.wheeltec_arc_actor and args.random_starts and args.cooperative_feedback)
  formation_motion_started_s=None
  first_motion_commands_s=[None]*len(states)
  formation_deadline_s=105. if synchronize_formation_start and args.communication=='nominal' else 90.
  if synchronize_formation_start:
   # Reserve time for dispersed Ackermann approach within the mission limit.
   formation_deadline_s=min(150.,max(0.,args.time_limit-45.))
   args.duration=max(args.duration,min(args.time_limit,formation_deadline_s+45.))
  if args.wheeltec_arc_actor:
   from cooperative_telemetry import CooperativeTelemetry
  cruise_speed=min(args.cooperative_cruise_speed,.32) if args.communication=='severe' else args.cooperative_cruise_speed

  if args.cooperative_feedback:
   from cooperative_pose_reference import (CooperativePoseReference, CooperativeTranslationReference,
                                           hold_pose_ready, hold_braking_risk, stopped)
   goals=[robot['target_world'] for robot in cooperative['samples'][0]['robots']]
   hold_destinations=cooperative.get('destinations',goals)
   hold_geometry_since=None
   area=json.loads((ROOT/'configs/tasks/mixed_formation_nominal080.json').read_text())['operating_area']
   wheeltec_settings={}
   if args.wheeltec_arc_actor or args.wheeltec_reversal_recovery:
    wheeltec_settings=dict(wheeltec_heading_tolerance_rad=math.radians(2.5),wheeltec_reversal_deceleration_mps2=.30)
   if args.wheeltec_hold_spacing:
    wheeltec_settings.update(wheeltec_heading_tolerance_rad=math.radians(3),
                             wheeltec_path_pair_tolerance_m=.075)
   cooperative_controller=CooperativePoseReference(goals,area,args.cooperative_maneuver_scale,
                                                   precision=args.wheeltec_hold_spacing,**wheeltec_settings)
   if args.wheeltec_arc_actor:
    import numpy as np
    artifact=json.loads(args.wheeltec_arc_actor.read_text())
    vertex_features=artifact.get('contract')=='wheeltec_vertex_arc_v1'
    weights={key:np.asarray(value,dtype=np.float32) for key,value in
             (artifact['weights'] if vertex_features else artifact).items()}
    if weights['0.weight'].shape[1]!=(10 if vertex_features else 7):
     raise ValueError('WheelTec actor input dimension does not match its contract.')
    def wheeltec_actor(features):
     nonlocal wheeltec_actor_calls
     wheeltec_actor_calls+=1
     hidden=np.tanh(weights['0.weight']@np.asarray(features,dtype=np.float32)+weights['0.bias'])
     return weights['2.weight']@hidden+weights['2.bias']
    def configure_wheeltec_actor(controller):
     controller.wheeltec_actor=wheeltec_actor
     controller.wheeltec_actor_uses_vertex=vertex_features
    configure_wheeltec_actor(cooperative_controller)
   if args.anchor_pose_actors:
    import numpy as np
    def load_anchor_actor(index,name):
     artifact=json.loads((args.anchor_pose_actors/f'{name}_actor.json').read_text())
     if artifact['contract']!='anchor_pose_v1' or artifact['robot']!=name:
      raise ValueError(f'Incompatible pose actor for {name}.')
     weights={key:np.asarray(value,dtype=np.float32) for key,value in artifact['weights'].items()}
     def infer(features):
      anchor_actor_calls[index]+=1
      hidden=np.tanh(weights['0.weight']@np.asarray(features,dtype=np.float32)+weights['0.bias'])
      return weights['2.weight']@hidden+weights['2.bias']
     return infer
    anchor_pose_actors={0:load_anchor_actor(0,'go2'),2:load_anchor_actor(2,'jackal')}
    cooperative_controller.anchor_pose_actors=anchor_pose_actors
   if synchronize_formation_start:
    cooperative_controller.path=list(plan['primitives'])
    cooperative_controller.segment_start=tuple(plan['start'])
   settled_ticks=0
  reference_index=0
  replans=0
  if args.reference_replan:
   from check_rosbot_formation_path import advance,find_path
   area=json.loads((ROOT/'configs/tasks/mixed_formation_nominal080.json').read_text())['operating_area']
  segment_start=states[1]
  reference_stop_time=None
  # Retain the validated paced bridge interval for moderate.
  short_bridge_pause=args.fast or (args.real_time_factor==.98 and args.communication!='moderate')
  bridge_pause_s=.012 if short_bridge_pause else .02
  physics_chunk_steps=int(os.environ.get('SWARM_PHYSICS_CHUNK_STEPS','20'))
  if physics_chunk_steps not in (20,50,100):
   raise ValueError('SWARM_PHYSICS_CHUNK_STEPS must be 20, 50 or 100.')
  with (out/'control.jsonl').open('w') as log:
   while True:
    if server.poll() is not None:raise RuntimeError('Gazebo server closed.')
    if gui is not None and gui.poll() is not None:done_reason='gui_closed';break
    commands=response['commands']
    control_states=states
    telemetry_ages=None
    # Own Go2 feedback remains local while neighbor telemetry uses the channel.
    pose_feedback=dict(go2_local_state=states[0])
    if args.communication=='severe' and telemetry is not None:
     pose_feedback=dict(go2_local_state=states[0],wheeltec_local_state=states[1],jackal_local_state=states[2])
    if telemetry is not None:
     control_states,telemetry_ages=telemetry.update(states,clock-origin)
    if control_states is None:
     commands=[[0.,0.]]*3
     can_pause_confirmation=(args.communication=='severe' and translation is None
                             and precision_settling and all(stopped(state) for state in states))
     if not can_pause_confirmation:
      settled_ticks=0
     hold_geometry_since=None
     if translation is None and clock-origin>=formation_deadline_s-1e-8:
      done_reason='cooperative_formation_timeout';break
    elif args.cooperative_feedback:
     if translation is None:
      if args.wheeltec_hold_spacing and hold_pose_ready(control_states,hold_destinations):
       commands=[[0.,0.]]*3
       status=None
       if all(stopped(state) for state in control_states):
        if hold_geometry_since is None:hold_geometry_since=clock
        if clock-hold_geometry_since>=2.-1e-8:
         done_reason='cooperative_pose_success';break
       else:hold_geometry_since=None
      else:
       hold_geometry_since=None
       commands,status=cooperative_controller.command(control_states,**pose_feedback)
       if args.wheeltec_hold_spacing and hold_braking_risk(control_states,commands[1],.30):
        commands[1]=[0.,0.]
        cooperative_controller.path.clear()
      pair_error=max(abs(math.hypot(a['x']-b['x'],a['y']-b['y'])-1.2)
                     for i,a in enumerate(control_states) for b in control_states[i+1:])
      pair_limit=.10
      if precision_started_s is not None:
       heading_error=max(abs(math.remainder(state['heading'],math.tau)) for state in control_states)
       precision_settling=(pair_error<=.075 and heading_error<=math.radians(5)
                           and (precision_settling or (pair_error<=.065 and heading_error<=math.radians(4))))
       pair_limit=.075
       if precision_settling:
        # Small odometry changes must not restart corrections while braking.
        commands=[[0.,0.]]*3
        status='settled' if all(stopped(state) for state in control_states) else None
      if status=='unresolved':
       done_reason='cooperative_pose_unresolved';break
      # Individually settled poses can still miss the pair-distance target.
      # Confirm rest before entering precision recovery, then enforce pair_limit.
      needs_precision=(args.cooperative_translation and args.go2_stride_cruise_only
                       and precision_started_s is None and pair_error>.065)
      settled_ticks=settled_ticks+1 if not args.wheeltec_hold_spacing and (status=='settled' or (status=='geometry_mismatch' and needs_precision)) and (pair_error<=pair_limit or needs_precision) else 0
      if settled_ticks>=21:
       if not args.cooperative_translation:
        done_reason='cooperative_pose_success';break
       if args.go2_stride_cruise_only and pair_error>.065 and precision_started_s is None:
        precision_started_s=clock-origin
        if args.wheeltec_arc_actor and args.communication=='nominal':
         formation_deadline_s=max(formation_deadline_s,min(110.,precision_started_s+20.))
        if args.communication=='severe':
         formation_deadline_s=max(formation_deadline_s,precision_started_s+25.)
         args.duration=max(args.duration,formation_deadline_s+45.)
        pose_fit=CooperativeTranslationReference(control_states)
        refined_goals=[(*vertex,0.) for vertex in pose_fit.translation_goals(0.)]
        cooperative_controller=CooperativePoseReference(refined_goals,area,args.cooperative_maneuver_scale,precision=True,**wheeltec_settings)
        if args.communication=='severe':
         for index in (0,2):
          state=states[index]
          goal=refined_goals[index]
          cooperative_controller.aligning[index]=math.hypot(state['x']-goal[0],state['y']-goal[1])<=.032
        if args.wheeltec_arc_actor:
         configure_wheeltec_actor(cooperative_controller)
        if args.anchor_pose_actors:
         cooperative_controller.anchor_pose_actors=anchor_pose_actors
        settled_ticks=0
       else:
        formation_completed_s=clock-origin
        translation=CooperativeTranslationReference(control_states,group_speed_mps=cruise_speed,adaptive_speed=args.cooperative_adaptive_speed,smooth_entry=args.go2_stride_cruise_only,terminal_recovery=args.cooperative_terminal_recovery)
        if args.translation_speed_actor:
         translation.speed_actor=translation_actor
        settled_ticks=0
      elif args.cooperative_translation and clock-origin>=formation_deadline_s-1e-8:
       done_reason='cooperative_formation_timeout';break
     if translation is not None:
      settled_ticks=settled_ticks+1 if not args.cooperative_terminal_recovery and translation.metrics(control_states)['settled'] else 0
      if settled_ticks>=21:
       done_reason='cooperative_mission_success';break
      commands=translation.command(control_states)
      if args.communication=='severe':
       # Keep catch-up commands within the same conservative speed envelope.
       commands=[[min(v,cruise_speed+.10),w] for v,w in commands]
      if args.go2_stride_cruise_only:
       braking_distance=cruise_speed**2/(2.*translation.braking_mps2)+.15
       progress=translation.reference_progress
       brake_start=translation.distance_m-braking_distance
       ramp_up=max(0.,min(1.,(progress-.25)/.70))
       ramp_down=max(0.,min(1.,(brake_start-progress)/.35))
       # Discrete parameter updates approximate a smooth change over distance.
       boost=round(6*min(ramp_up,ramp_down))/6.
       next_scale=round(1.+boost*(args.go2_forward_stride_scale-1.),6)
       if next_scale!=stride_scale:
        if stride_control is None:stride_control=Go2StrideControl()
        stride_control.set(next_scale)
        stride_scale=next_scale
        stride_changes.append(dict(time_s=clock-origin,reference_progress_m=translation.reference_progress,scale=stride_scale))

      if args.cooperative_terminal_recovery:
       terminal_goals=[(*point,0.) for point in translation.goals]
       if (terminal_controller is None and translation.reference_progress>=translation.distance_m-.08
           and all(stopped(state) for state in control_states)
           # Leave room to recover overshoot after braking with delayed odometry.
           and translation.metrics(control_states)['max_destination_error_m']<=.40):
        terminal_controller=CooperativePoseReference(
            terminal_goals,area,precision=True,wheeltec_heading_tolerance_rad=math.radians(3),
            wheeltec_reversal_deceleration_mps2=.30,wheeltec_path_pair_tolerance_m=.075,
            wheeltec_goal_pair_tolerance_m=terminal_pair_tolerance_m)
        if args.wheeltec_arc_actor and vertex_features:
         configure_wheeltec_actor(terminal_controller)
        if args.anchor_pose_actors:
         terminal_controller.anchor_pose_actors=anchor_pose_actors
        terminal_started_s=clock-origin
        # The Ackermann recovery can require multiple forward/reverse arcs.
        # Allocate its own budget within the configured mission deadline.
        args.duration=max(args.duration,min(args.time_limit,terminal_started_s+90.))
       if terminal_controller is not None:
        if hold_pose_ready(control_states,terminal_goals,pair_tolerance_m=terminal_pair_tolerance_m):
         commands=[[0.,0.]]*3
         if all(stopped(state) for state in control_states):
          if hold_geometry_since is None:hold_geometry_since=clock
          if (clock-hold_geometry_since>=2.-1e-8
              and hold_pose_ready(states,terminal_goals,pair_tolerance_m=terminal_pair_tolerance_m)
              and all(stopped(state) for state in states)):
           done_reason='cooperative_mission_success';break
         else:hold_geometry_since=None
        else:
         hold_geometry_since=None
         commands,status=terminal_controller.command(control_states,**pose_feedback)
         if status=='unresolved':
          done_reason='cooperative_terminal_unresolved';break
         braking_states=control_states
         if 'wheeltec_local_state' in pose_feedback:
          braking_states=[control_states[0],pose_feedback['wheeltec_local_state'],control_states[2]]
         if hold_braking_risk(braking_states,commands[1],.30,recovery_slack_m=.02 if 'wheeltec_local_state' in pose_feedback else 0.):
          commands[1]=[0.,0.]
          terminal_controller.path.clear()
          terminal_brake_ticks+=1
         terminal_correction_ticks+=int(any(v or w for v,w in commands))

    elif cooperative:
     commands=[robot['command'] for robot in cooperative['samples'][tick]['robots']]
    elif reference:
     index=reference_index if args.reference_feedback else tick//12
     command=[0.,0.]
     reached=False
     if args.reference_feedback and index<len(reference['primitives']):
      travel,curvature=reference['primitives'][index]
      own=states[1]
      if curvature:
       target_turn=travel*curvature
       measured_turn=math.remainder(own['heading']-segment_start['heading'],2*math.pi)
       reached=math.copysign(1.,target_turn)*measured_turn>=abs(target_turn)-.004
      else:
       dx=own['x']-segment_start['x']
       dy=own['y']-segment_start['y']
       along=dx*math.cos(segment_start['heading'])+dy*math.sin(segment_start['heading'])
       reached=math.copysign(1.,travel)*along>=abs(travel)-.003
      if reached:
       reference_index+=1
       index=reference_index
       previous_segment=segment_start
       segment_start=own
     if args.reference_replan:
      own=states[1]
      pose=(own['x'],own['y'],own['heading'])
      anchors=[(states[i]['x'],states[i]['y']) for i in (0,2)]
      position_ok=math.dist(pose[:2],reference['goal'])<=.08
      heading_ok=abs(pose[2])<=math.radians(5)
      pair_error=max(abs(math.dist((a['x'],a['y']),(b['x'],b['y']))-1.2)
                     for i,a in enumerate(states) for b in states[i+1:])
      if min(math.dist(pose[:2],anchor) for anchor in anchors)<.60:
       done_reason='reference_clearance_limit';break
      if position_ok and heading_ok and pair_error<=.10:
       reference_index=index=len(reference['primitives'])
      elif index>=len(reference['primitives']):
       if abs(own['speed'])<=.03 and abs(own['yaw_rate'])<=.05:
        result=find_path(pose,reference['goal'],anchors,area)
        if result is None or not result[0]:
         done_reason='reference_pose_unresolved';break
        reference['primitives']=result[0]
        reference_index=index=0
        segment_start=own
        reference_stop_time=None
        replans+=1
      elif reached:
       predicted=advance((previous_segment['x'],previous_segment['y'],previous_segment['heading']),travel,curvature)
       if math.dist(pose[:2],predicted[:2])>.06:
        # Stop before planning from a new pose.
        reference_index=index=len(reference['primitives'])
     if index<len(reference['primitives']):
      travel,curvature=reference['primitives'][index]
      speed=math.copysign(.15,travel)
      command=[speed,speed*curvature]
     elif args.reference_feedback and reference_stop_time is None:
      reference_stop_time=clock-origin
     commands=[[0.,0.],command,[0.,0.]]
    elif args.cruise_actuator_probe:
     if args.go2_speed_sweep:
      # Six seconds per command, followed by a three-second stop.
      speeds=(.30,.45,.60,.80,1.00,1.30,1.60,0.)
      commands=[[speeds[min(tick//60,7)],0.],[0.,0.],[0.,0.]]
     else:
      commands=[[1.60 if tick<70 else 0.,0.] for _ in range(3)]
    elif args.mixed_actuator_probe:
     # Small command, straight advance, turn, then brake and settle.
     elapsed=tick*.1
     if elapsed<2.:command=[.012,0.]
     elif elapsed<5.:command=[.30,0.]
     elif elapsed<8.:command=[.30,.40]
     else:command=[0.,0.]
     commands=[command.copy() for _ in range(3)]
    elif args.probe:
     end=args.duration or 5.;commands=[[.1 if tick*.1<end*.6 else 0.,0.],[0.,0.],[0.,0.]]
    if args.mixed:
     from swarm_core.mixed_commands import constrain_mixed_commands
     commands=constrain_mixed_commands(commands)
    ramp_go2=(args.mixed and args.cooperative_feedback and translation is not None
              and terminal_controller is None)
    if ramp_go2:
     # Limit forward acceleration at the 10 Hz command rate. Use the last
     # applied command so a clearance stop cannot accumulate a hidden ramp.
     speed,yaw=commands[0]
     if speed>0.:
      commands=[list(command) for command in commands]
      commands[0]=[min(speed,1.05,previous_go2_forward+.08),yaw]
    go2_clearance_limited=False
    if telemetry is not None:
     commands=telemetry.constrain_commands(control_states,telemetry_ages,commands)
    if args.wheeltec_arc_actor:
     filtered_commands=CooperativeTelemetry.filter_local_clearance(states,commands)
     clearance_filtered=filtered_commands!=commands
     go2_clearance_limited=filtered_commands[0]!=commands[0]
     local_clearance_stops+=int(clearance_filtered)
     if (translation is None and formation_motion_started_s is not None
         and control_states is not None):
      if clearance_filtered and not any(v or w for v,w in filtered_commands):
       clearance_blocked_ticks+=1
       if clearance_blocked_ticks>=3:
        # Discard the obstructed arc and plan around the same margin as the stop filter.
        cooperative_controller.path.clear()
        cooperative_controller.plan_cooldown=0
        cooperative_controller.failed_plan_anchors=None
        cooperative_controller.wheeltec_replan_margin_m=.70
        cooperative_controller.wheeltec_clearance_recovery=True
        formation_clearance_replans+=1
        clearance_blocked_ticks=0
      else:
       clearance_blocked_ticks=0
     commands=filtered_commands
    if synchronize_formation_start and formation_motion_started_s is None:
     # A robot already at its aligned vertex can participate without moving.
     ready=all(v or w or (math.hypot(state['x']-goal[0],state['y']-goal[1])<=.04
                          and abs(math.remainder(state['heading']-goal[2],math.tau))<=.035)
               for (v,w),state,goal in zip(commands,states,goals))
     if ready and control_states is not None:
      formation_motion_started_s=clock-origin
     else:
      # Anchors already at their vertex may need yaw alignment before the
      # Ackermann planner can release its first arc.
      commands=[[0., w] if i in (0,2) and not v
                and math.hypot(state['x']-goal[0],state['y']-goal[1])<=.05
                else [0.,0.]
                for i,((v,w),state,goal) in enumerate(zip(commands,states,goals))]
    if synchronize_formation_start:
     for i,(v,w) in enumerate(commands):
      if first_motion_commands_s[i] is None and (v or w):
       first_motion_commands_s[i]=clock-origin
    previous_go2_forward=max(0.,commands[0][0]) if ramp_go2 else 0.
    transport.commands(commands)
    if args.mixed:
     for _ in range(100//physics_chunk_steps):
      clock,states,ages=transport.advance(physics_chunk_steps)
      time.sleep(bridge_pause_s)
    else:
     clock,states,ages=transport.advance(20)
    tick+=1;elapsed=clock-origin
    if worker:response=rpc(dict(op='step',states=states,clock_s=clock))
    else:response=dict(commands=commands,modes=['probe']*3,time_s=elapsed,done=False,reason=None)
    if args.cooperative_feedback and translation is None:
     formation_error=max(abs(math.hypot(a['x']-b['x'],a['y']-b['y'])-1.2)
                         for i,a in enumerate(states) for b in states[i+1:])
     response.update(max_pair_error_m=formation_error,
                     modes=['cooperative_precision' if precision_started_s is not None else 'cooperative_formation']*3)
    if translation is not None:
     response.update(translation.metrics(states),modes=['cooperative_terminal_recovery' if terminal_controller is not None else 'cooperative_translation']*3)
    if telemetry is not None:response['cooperative_telemetry_age_s']=telemetry_ages
    log.write(json.dumps(dict(clock_s=clock,elapsed_s=elapsed,states=states,odometry_age_s=ages,applied_commands=commands,go2_clearance_limited=go2_clearance_limited,policy=response,**({'reference_primitive':index} if reference else {})))+chr(10));log.flush()
    if int(elapsed)//5!=last_report:
     last_report=int(elapsed)//5;print(f"sim={elapsed:.1f}s modes={response['modes']} valid={response.get('valid_neighbors')} ready={response.get('startup_ready_steps')} pair_error={response.get('max_pair_error_m')} RTF={elapsed/max(time.monotonic()-started,1e-9):.2f}",flush=True)
    if cooperative:
     clearance=min(math.hypot(a['x']-b['x'],a['y']-b['y'])-.50
                   for i,a in enumerate(states) for b in states[i+1:])
     if clearance<(.15 if args.wheeltec_arc_actor else .10)-1e-9:
      done_reason='cooperative_clearance_limit';break
    if args.mixed and (states[0]['z']<.13 or states[0]['up_z']<.65):
     done_reason='go2_fall';break
    if response.get('mission_completed') and not success_reported:
     success_reported=True;(out/'mission_success.json').write_text(json.dumps(response,indent=2));print('Mission completed at',response['first_success_s'],'simulation seconds. Live hold:',not response['done'],flush=True)
    if not args.fast:time.sleep(max(0.,started+elapsed/args.real_time_factor-time.monotonic()))
    if reference_stop_time is not None and elapsed-reference_stop_time>=3.-1e-8 and (not args.reference_replan or (position_ok and heading_ok and pair_error<=.10)):
     done_reason='reference_completed';break
    limit=args.duration or (5. if args.probe else args.time_limit)
    timed_out=elapsed>=limit-1e-8 and (args.duration is not None or args.probe or not response.get('mission_completed'))
    if response.get('done') or timed_out:done_reason=response.get('reason') if response.get('done') else 'duration_limit';break
  transport.commands([[0.,0.]]*3);transport.control()
  displacement=[math.hypot(s['x']-i['x'],s['y']-i['y']) for s,i in zip(states,initial)]
  summary=dict(reason=done_reason,simulated_s=clock-origin,wall_s=time.monotonic()-started,physical_displacement_m=displacement,final_policy=response,probe=args.probe,feedback_steps=tick,target_real_time_factor=None if args.fast else args.real_time_factor)
  if args.cooperative_feedback:
   summary['replans']=cooperative_controller.replans
   summary['cooperative_maneuver_scale']=args.cooperative_maneuver_scale
  if args.cooperative_translation:
   summary['formation_completed_s']=formation_completed_s
   summary['precision_started_s']=precision_started_s
   summary['cooperative_cruise_speed_mps']=cruise_speed
   summary['cooperative_adaptive_speed']=args.cooperative_adaptive_speed
   if translation is not None:summary['translation']=translation.metrics(states)
   if args.go2_stride_cruise_only:summary['go2_stride_changes']=stride_changes
  if args.reference_replan:summary['replans']=replans
  if args.mixed:summary['go2_forward_stride_scale']=args.go2_forward_stride_scale
  if args.cooperative_terminal_recovery:
   summary.update(terminal_started_s=terminal_started_s,terminal_correction_ticks=terminal_correction_ticks,terminal_brake_ticks=terminal_brake_ticks)
  if args.wheeltec_arc_actor:
   summary.update(wheeltec_actor_calls=wheeltec_actor_calls,local_clearance_stops=local_clearance_stops,formation_clearance_replans=formation_clearance_replans)
  if synchronize_formation_start:
   summary.update(formation_motion_started_s=formation_motion_started_s,
                  first_motion_commands_s=first_motion_commands_s)
  if args.anchor_pose_actors:summary['anchor_actor_calls']=anchor_actor_calls
  if args.translation_speed_actor:summary['translation_actor_calls']=translation_actor_calls
  if telemetry is not None:summary['cooperative_telemetry']=telemetry.metrics()
  summary['measured_real_time_factor']=summary['simulated_s']/max(summary['wall_s'],1e-9)
  summary['physics_chunk_steps']=physics_chunk_steps if args.mixed else None
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
  if gui is not None and not args.no_keep_open:
   print('Episode stopped. Gazebo remains open; close its window or Ctrl+C to exit.',flush=True)
   while gui.poll() is None and server.poll() is None:time.sleep(.2)
 finally:
  if stride_control is not None:stride_control.close()
  if transport is not None:
   try:transport.commands([[0.,0.]]*3);transport.control(timeout=200)
   except Exception:pass
  for child in reversed(children):
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()
  for handle in handles:handle.close()

if __name__=='__main__':
 try:main()
 except KeyboardInterrupt:pass
