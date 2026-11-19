"""Add visual-only destination markers to the owned trajectory replay world."""
import argparse,json,os,math
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--partition',required=True);parser.add_argument('--goals',required=True);args=parser.parse_args()
 os.environ['GZ_PARTITION']=args.partition
 from gz.transport13 import Node
 from gz.msgs10.entity_factory_pb2 import EntityFactory
 from gz.msgs10.boolean_pb2 import Boolean
 data=json.loads(Path(args.goals).read_text());goals=data['goals'];colors=['0.2 0.7 1 1','1 0.65 0.15 1','0.7 0.3 1 1'];visuals=[]
 for i,(x,y) in enumerate(goals):
  visuals.append(f'<visual name="goal_{i}"><pose>{x} {y} 0.015 0 0 0</pose><geometry><cylinder><radius>0.32</radius><length>0.012</length></cylinder></geometry><material><ambient>{colors[i]}</ambient><diffuse>{colors[i]}</diffuse></material><transparency>0.3</transparency><cast_shadows>false</cast_shadows></visual>')
 for i,j in [(0,1),(1,2),(2,0)]:
  a,b=goals[i],goals[j];dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);angle=math.atan2(dy,dx)
  visuals.append(f'<visual name="edge_{i}_{j}"><pose>{(a[0]+b[0])/2} {(a[1]+b[1])/2} 0.02 0 0 {angle}</pose><geometry><box><size>{length} 0.025 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material><cast_shadows>false</cast_shadows></visual>')
 request=EntityFactory();request.sdf='<sdf version="1.9"><model name="mission_destinations"><static>true</static><link name="visual_markers">'+''.join(visuals)+'</link></model></sdf>'
 node=Node();success,response=node.request('/world/swarm_replay/create',request,EntityFactory,Boolean,3000)
 if not success or not response.data:raise RuntimeError('Could not create goal markers in the requested replay world.')
 print('Destination markers created. Colors match robots; white edges show the target triangle.')

if __name__=='__main__':main()
