"""Start Gazebo server, GUI and replay with readiness checks and cleanup."""
import argparse,os,subprocess,sys,time
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

def main():
 parser=argparse.ArgumentParser();parser.add_argument('run',nargs='?',default='experiments/runs/formation_proximity_001');parser.add_argument('--speed',default='2');parser.add_argument('--gui-config');parser.add_argument('--once',action='store_true');parser.add_argument('--end-hold-s',default='0');args=parser.parse_args()
 root=Path(__file__).resolve().parents[1];os.chdir(root)
 if not (Path(args.run)/'trajectory.json').exists():parser.error('Recorded trajectory not found.')
 logs=root/'experiments/runs/gazebo_viewer';logs.mkdir(exist_ok=True)
 processes=[];handles=[]
 # Isolate this viewer from any other Gazebo worlds or terminals.
 os.environ['GZ_PARTITION']='swarm_replay_'+str(os.getpid())
 try:
  for name,command in [('server',['gz','sim','-s','-r',str(root/'gazebo/replay.sdf')]),('gui',['gz','sim','-g','--gui-config',str(Path(args.gui_config).resolve() if args.gui_config else root/'gazebo/replay_gui.config')])]:
   handle=(logs/(name+'.log')).open('w');handles.append(handle)
   processes.append(subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT))
  node=Node();ready=False
  for _ in range(60):
   if processes[0].poll() is not None:break
   pose=Pose();pose.name='robot_0';pose.position.z=.25;pose.orientation.w=1
   success,response=node.request('/world/swarm_replay/set_pose',pose,Pose,Boolean,500)
   if success and response.data:ready=True;break
   time.sleep(.25)
  if not ready:raise RuntimeError('Gazebo server did not become ready. See '+str(logs/'server.log'))
  print('Gazebo world ready. GUI log: '+str(logs/'gui.log'),flush=True)
  print('Recorded trajectory playback; Ctrl+C stops all viewer processes.',flush=True)
  goal_file=Path(args.run)/'mission_goals.json'
  if goal_file.exists():subprocess.run([sys.executable,str(root/'scripts/add_gazebo_goals.py'),'--partition',os.environ['GZ_PARTITION'],'--goals',str(goal_file)],check=True)
  replay_command=[sys.executable,str(root/'scripts/replay_gazebo.py'),args.run,'--speed',args.speed,'--end-hold-s',args.end_hold_s]
  if args.once:replay_command.append('--once')
  replay=subprocess.Popen(replay_command);processes.append(replay)
  while replay.poll() is None:
   if processes[0].poll() is not None or processes[1].poll() is not None:
    raise RuntimeError('Gazebo closed. Consult server.log and gui.log in '+str(logs))
   time.sleep(.25)
  if replay.returncode:raise RuntimeError('Playback failed; exit code '+str(replay.returncode))
 except KeyboardInterrupt:pass
 finally:
  for process in reversed(processes):
   if process.poll() is None:
    process.terminate()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.kill();process.wait()
  for handle in handles:handle.close()

if __name__=='__main__':main()
