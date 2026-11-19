"""Replay recorded poses in Gazebo; this is not a physics validation."""
import argparse,json,math,time
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean

def set_frame(node,frame):
 for i,(x,y,heading) in enumerate(frame):
  pose=Pose();pose.name=f'robot_{i}';pose.position.x=x;pose.position.y=y;pose.position.z=.25
  pose.orientation.z=math.sin(heading/2);pose.orientation.w=math.cos(heading/2)
  success,response=node.request('/world/swarm_replay/set_pose',pose,Pose,Boolean,1000)
  if not success or not response.data:raise RuntimeError('Gazebo pose service unavailable; start gazebo/replay.sdf first.')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--speed',type=float,default=1.);parser.add_argument('--once',action='store_true');parser.add_argument('--first-frame',action='store_true');parser.add_argument('--end-hold-s',type=float,default=0.);cli=parser.parse_args()
 if not math.isfinite(cli.speed) or cli.speed<=0:parser.error('Speed must be positive.')
 if not math.isfinite(cli.end_hold_s) or cli.end_hold_s<0:parser.error('Display hold must be finite and nonnegative.')
 run=Path(cli.run);frames=json.loads((run/'trajectory.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
 period=manifest['task']['simulation']['time_step_s']/cli.speed
 node=Node();time.sleep(.5)
 print('Recorded Python trajectory: NOT Gazebo physics or validated robot control.',flush=True)
 while True:
  start=time.monotonic()
  for index,frame in enumerate(frames):
   set_frame(node,frame)
   if cli.first_frame:print('Gazebo pose delivery: PASSED');return
   time.sleep(max(0.,start+(index+1)*period-time.monotonic()))
  if cli.end_hold_s:
   print('End of recorded episode: display-only final-frame pause.',flush=True)
   time.sleep(cli.end_hold_s)
  if cli.once:return
  print('Restarting recorded trajectory.',flush=True)

if __name__=='__main__':
 try:main()
 except KeyboardInterrupt:pass
