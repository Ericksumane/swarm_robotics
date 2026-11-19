"""Full local mission ROS adapter. Outputs only a shadow velocity topic.

No vendor driver or actuator enabling is provided. All robot poses must already
be in the configured common frame; this node does not invent frame alignment.
"""
import copy,hashlib,json,math,os,select,subprocess,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from swarm_interfaces.msg import PeerState

def seconds(stamp):return stamp.sec+stamp.nanosec/1e9
def set_stamp(stamp,value):
    total=round(value*1e9);stamp.sec=total//1000000000;stamp.nanosec=total%1000000000

def peer_dict(msg):
    return dict(protocol_version=int(msg.protocol_version),mission_id=msg.mission_id,session_id=msg.session_id,frame_id=msg.header.frame_id,faulted=bool(msg.faulted),state=dict(sender_id=int(msg.sender_id),sequence=int(msg.sequence),generated_at_s=seconds(msg.header.stamp),x=msg.x,y=msg.y,heading=msg.heading,speed=msg.speed,yaw_rate=msg.yaw_rate,payload_size_bytes=72,formation_ready=bool(msg.formation_ready),start_at_s=msg.start_at_s if msg.has_start_proposal else None,start_committed=bool(msg.start_committed)))

def peer_message(value):
    msg=PeerState();s=value['state'];msg.protocol_version=value['protocol_version'];msg.mission_id=value['mission_id'];msg.session_id=value['session_id'];msg.header.frame_id=value['frame_id'];set_stamp(msg.header.stamp,s['generated_at_s'])
    for key in ('sender_id','sequence','x','y','heading','speed','yaw_rate','formation_ready','start_committed'):setattr(msg,key,s[key])
    msg.has_start_proposal=s['start_at_s'] is not None;msg.start_at_s=s['start_at_s'] if msg.has_start_proposal else 0.;msg.faulted=value['faulted'];return msg

class FullMissionNode(Node):
    def __init__(self):
        super().__init__('full_mission')
        defaults=dict(project_root='',deployment_manifest='',robot_id=-1,mission_id='',common_frame='mission',body_frame='base_link',completion_profile='nominal',rpc_timeout_s=.2)
        for name,value in defaults.items():self.declare_parameter(name,value)
        get=lambda key:self.get_parameter(key).value
        self.robot_id=get('robot_id');self.frame=get('common_frame');self.body=get('body_frame');self.timeout=get('rpc_timeout_s')
        if self.robot_id not in (0,1,2) or not get('mission_id') or not get('project_root'):raise ValueError('Set project_root, robot_id and a common unique mission_id.')
        if not math.isfinite(self.timeout) or self.timeout<=0:raise ValueError('Invalid RPC timeout.')
        root=Path(get('project_root')).resolve();manifest_path=Path(get('deployment_manifest')) if get('deployment_manifest') else root/'configs/deployment/delivery_candidate.json'
        manifest=json.loads(manifest_path.read_text());bundle=root/manifest['actor_bundle']
        if manifest.get('supported_team_size')!=3:raise ValueError('Unsupported team size.')
        self.commands=self.create_publisher(Twist,'policy/cmd_vel',10);self.status=self.create_publisher(String,'policy/status',10)
        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)
        self.peer_pub=self.create_publisher(PeerState,f'/swarm/robot_{self.robot_id}/state_v1',qos)
        self.own=None;self.own_stamp=None;self.own_wall=None;self.pending={};self.fault=None;self.last_peer=None;self.previous_mode=None;self.last_phase=0;self.last_control_wall=time.monotonic()
        self.process=None;self.stderr=None
        logs=root/'experiments/runtime'/hashlib.sha256(get('mission_id').encode()).hexdigest()[:16];logs.mkdir(parents=True,exist_ok=True)
        self.stderr=(logs/f'robot_{self.robot_id}_{os.getpid()}.log').open('w')
        env=os.environ.copy()
        for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):env.pop(key,None)
        command=[str(root/'.venv/bin/python'),str(root/'scripts/robot_mission_service.py'),'--bundle',str(bundle),'--expected-sha256',manifest['actor_bundle_sha256'],'--robot-id',str(self.robot_id),'--mission-id',get('mission_id'),'--frame-id',self.frame,'--profile',get('completion_profile')]
        try:
            self.process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
            hello=self.receive(60.)
            if hello.get('kind')!='ready' or hello.get('bundle_sha256')!=manifest['actor_bundle_sha256']:raise RuntimeError('Invalid mission service handshake.')
            self.create_subscription(Odometry,'odom',self.odometry,qos_profile_sensor_data)
            self.peers=[self.create_subscription(PeerState,f'/swarm/robot_{i}/state_v1',self.peer,qos) for i in range(3) if i!=self.robot_id]
            self.timer=self.create_timer(.1,self.tick)
            self.wall_timer=self.create_timer(.1,self.watchdog,clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.last_control_wall=time.monotonic()
            self.add_on_set_parameters_callback(lambda parameters:SetParametersResult(successful=False,reason='Mission configuration is immutable; restart all nodes with a new mission ID.'))
            self.get_logger().info('Full local mission ready. SHADOW OUTPUT ONLY: policy/cmd_vel. No hardware adapter enabled.')
        except Exception:
            self.close_process();raise

    def receive(self,timeout):
        if self.process is None or not select.select([self.process.stdout],[],[],timeout)[0]:raise TimeoutError('Mission service timeout.')
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Mission service exited.')
        result=json.loads(line)
        if result.get('kind')=='error':raise RuntimeError(result.get('error','Mission service error.'))
        return result

    def rpc(self,request):
        self.process.stdin.write(json.dumps(dict(protocol_version=1,**request),allow_nan=False)+'\n');self.process.stdin.flush();return self.receive(self.timeout)

    def odometry(self,msg):
        try:
            if msg.header.frame_id!=self.frame or msg.child_frame_id!=self.body:raise ValueError('Odometry frame mismatch.')
            p=msg.pose.pose.position;q=msg.pose.pose.orientation;v=msg.twist.twist
            values=[p.x,p.y,q.x,q.y,q.z,q.w,v.linear.x,v.linear.y,v.angular.z]
            if not all(math.isfinite(x) for x in values) or abs(sum(x*x for x in (q.x,q.y,q.z,q.w))-1.)>.001:raise ValueError('Invalid odometry.')
            if abs(v.linear.y)>.03:raise ValueError('Lateral motion exceeds the synthetic nonholonomic profile.')
            stamp=seconds(msg.header.stamp)
            if self.own_stamp is not None and stamp<=self.own_stamp:return
            self.own=dict(x=p.x,y=p.y,heading=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),speed=v.linear.x,yaw_rate=v.angular.z)
            self.own_stamp=stamp;self.own_wall=time.monotonic()
        except Exception as error:self.fail(str(error))

    def peer(self,msg):
        sender=int(msg.sender_id)
        if sender==self.robot_id or sender not in (0,1,2):return
        previous=self.pending.get(sender)
        if previous is None or msg.session_id!=previous.session_id or msg.sequence>previous.sequence or msg.faulted:self.pending[sender]=msg

    def fail(self,reason):
        if self.fault is None:self.fault=reason;self.get_logger().error('Latched stop: '+reason)
        self.commands.publish(Twist())
        status=String();status.data=json.dumps(dict(shadow_only=True,robot_id=self.robot_id,mode='fault_stop',phase=self.last_phase,fault=self.fault,command=[0.,0.]));self.status.publish(status)
        if self.last_peer is not None:
            packet=copy.deepcopy(self.last_peer);packet['faulted']=True;packet['state']['formation_ready']=False;packet['state']['start_committed']=False;packet['state']['start_at_s']=None;packet['state']['sequence']+=1;packet['state']['generated_at_s']=self.get_clock().now().nanoseconds/1e9
            self.peer_pub.publish(peer_message(packet))

    def tick(self):
        self.last_control_wall=time.monotonic()
        if self.fault:self.fail(self.fault);return
        if self.own is None:self.commands.publish(Twist());return
        try:
            now=self.get_clock().now().nanoseconds/1e9;started=time.monotonic()
            pending=self.pending;self.pending={}
            for msg in pending.values():self.rpc(dict(op='peer',peer=peer_dict(msg),now_s=now))
            result=self.rpc(dict(op='tick',own_state=self.own,now_s=now,odometry_stamp_s=self.own_stamp))
            if result.get('kind')!='tick' or time.monotonic()-started>self.timeout:raise RuntimeError('Invalid or late mission tick.')
            self.last_phase=result['phase'];self.last_peer=result['peer'];self.peer_pub.publish(peer_message(self.last_peer))
            if result['fault']:self.fail(result['fault']);return
            v,w=result['command']
            if not all(math.isfinite(x) for x in (v,w)):raise ValueError('Nonfinite command.')
            message=Twist();message.linear.x=v;message.angular.z=w;self.commands.publish(message)
            status=String();status.data=json.dumps(dict(shadow_only=True,robot_id=self.robot_id,tick_compute_s=time.monotonic()-started,**result),allow_nan=False);self.status.publish(status)
            if result['mode']!=self.previous_mode:self.get_logger().info('Mode: '+result['mode']);self.previous_mode=result['mode']
        except Exception as error:self.fail(str(error));self.close_process()

    def watchdog(self):
        now=time.monotonic()
        if now-self.last_control_wall>.5 or (self.own_wall is not None and now-self.own_wall>.5):self.fail('Wall-clock control/odometry watchdog.')

    def close_process(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=5.)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process=None
        if self.stderr is not None:self.stderr.close();self.stderr=None

    def destroy_node(self):
        try:self.commands.publish(Twist())
        except Exception:pass
        self.close_process();return super().destroy_node()

def main(args=None):
    rclpy.init(args=args);node=None
    try:node=FullMissionNode();rclpy.spin(node)
    except (KeyboardInterrupt,ExternalShutdownException):pass
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
