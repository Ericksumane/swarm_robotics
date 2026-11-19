"""Three real ROS processes: peer protocol, scheduled start and fault stop.

Odometry is a stationary triangle fixture. No actuator or physics is connected;
this smoke test validates ROS transport/control, not trajectory completion.
"""
import argparse,json,math,os,signal,subprocess,time,uuid
from pathlib import Path
os.environ['ROS_DOMAIN_ID']=str(80+os.getpid()%100)
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root');parser.add_argument('--output');parser.add_argument('--manifest');cli=parser.parse_args()
    root=Path(cli.project_root).resolve() if cli.project_root else Path(__file__).resolve().parents[1]
    out=Path(cli.output) if cli.output else root/('experiments/runs/delivery_completion_001/ros_shadow_smoke_'+str(os.getpid()))
    out.mkdir(parents=True,exist_ok=False)
    executable=root/'ros2_ws/install/swarm_runtime/lib/swarm_runtime/full_mission_node'
    manifest=Path(cli.manifest) if cli.manifest else root/'configs/deployment/delivery_candidate.json'
    processes=[];logs=[];rclpy.init();node=Node('shadow_test_fixture');statuses={};commands={};starts={};source_starts={};latest_latencies=[]
    publishers=[node.create_publisher(Odometry,f'/robot_{i}/odom',10) for i in range(3)]
    def status(i,msg):
        value=json.loads(msg.data);statuses[i]=value
        if value.get('phase',0)>=1 and i not in starts:starts[i]=time.monotonic();source_starts[i]=value.get('move_started_at_s')
        if 'tick_compute_s' in value:latest_latencies.append(value['tick_compute_s'])
    subscribers=[]
    for i in range(3):
        subscribers.append(node.create_subscription(String,f'/robot_{i}/policy/status',lambda msg,i=i:status(i,msg),10))
        subscribers.append(node.create_subscription(Twist,f'/robot_{i}/policy/cmd_vel',lambda msg,i=i:commands.update({i:[msg.linear.x,msg.angular.z]}),10))
    mission='shadow_'+uuid.uuid4().hex;positions=[(0.,0.),(1.,0.),(.5,math.sqrt(3)/2)];started=time.monotonic();last_publish=0.;stop_odom=False;fault_start=None
    try:
        for i in range(3):
            stream=(out/f'robot_{i}.log').open('w');logs.append(stream)
            args=[str(executable),'--ros-args','-r',f'__ns:=/robot_{i}','-p',f'project_root:={root}','-p',f'deployment_manifest:={manifest}','-p',f'robot_id:={i}','-p',f'mission_id:={mission}','-p','common_frame:=mission','-p',f'body_frame:=robot_{i}/base_link']
            processes.append(subprocess.Popen(args,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while time.monotonic()-started<65.:
            now=time.monotonic()
            if now-last_publish>=.05:
                last_publish=now
                for i,(x,y) in enumerate(positions):
                    if stop_odom and i==1:continue
                    msg=Odometry();msg.header.stamp=node.get_clock().now().to_msg();msg.header.frame_id='mission';msg.child_frame_id=f'robot_{i}/base_link';msg.pose.pose.position.x=x;msg.pose.pose.position.y=y;msg.pose.pose.orientation.w=1.;publishers[i].publish(msg)
            rclpy.spin_once(node,timeout_sec=.01)
            if any(p.poll() is not None for p in processes):raise AssertionError('A mission node exited; inspect robot logs.')
            if len(starts)==3 and not stop_odom:
                assert all(statuses[i]['goal_m'] is not None and abs(statuses[i]['goal_m'][0]-positions[i][0]-5.)<1e-7 for i in range(3))
                if now-max(starts.values())>2.:
                    assert any(abs(v[0])>.01 for v in commands.values()),commands
                    stop_odom=True;fault_start=now
            if stop_odom and now-fault_start>2.:
                assert all(statuses.get(i,{}).get('fault') for i in range(3)),statuses
                assert all(abs(v[0])+abs(v[1])<1e-9 for v in commands.values()),commands
                values=sorted(latest_latencies)
                result=dict(scope='Three independent ROS shadow nodes; static odometry, no physical motion',mission_id=mission,domain_id=os.environ['ROS_DOMAIN_ID'],scheduled_start_observation_skew_s=max(starts.values())-min(starts.values()),source_move_start_s=source_starts,source_start_skew_s=max(source_starts.values())-min(source_starts.values()),max_tick_compute_s=max(values),p99_tick_compute_s=values[min(len(values)-1,int(.99*len(values)))],faults={i:statuses[i]['fault'] for i in range(3)},commands_after_fault=commands,result='passed')
                (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
        raise AssertionError('ROS shadow scenario did not finish in 65 seconds: '+str(statuses))
    finally:
        for process in processes:
            if process.poll() is None:os.killpg(process.pid,signal.SIGINT)
        for process in processes:
            try:process.wait(timeout=10.)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        for stream in logs:stream.close()
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
