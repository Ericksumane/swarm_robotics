"""Local ROS shadow actor: odometry and peer states in, desired Twist out.

Output is policy/cmd_vel and must not be remapped to physical actuators
before platform calibration, watchdog and independent stop validation.
All odometry inputs must already use a common frame and synchronized time.
"""
import json,math,select,subprocess,sys,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

class PolicyNode(Node):
 def __init__(self):
  super().__init__('formation_policy')
  for name,value in [('project_root','.'),
      ('artifact',''),('robot_id',0),('num_agents',3),('max_age_s',0.5)]:
   self.declare_parameter(name,value)
  get=lambda key:self.get_parameter(key).value
  self.root=Path(get('project_root'));self.robot_id=get('robot_id');self.n=get('num_agents')
  self.max_age=get('max_age_s');self.state=None;self.received={};self.seq=0
  self.task=json.loads((self.root/'configs/tasks/formation_v1.json').read_text())
  self.limits=self.task['robot_defaults'];self.last_command=(0.,0.)
  self.process=None
  if get('artifact'):
   self.process=subprocess.Popen([str(self.root/'.venv/bin/python'),str(self.root/'scripts/actor_service.py'),get('artifact')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
  self.publisher=self.create_publisher(Twist,'policy/cmd_vel',10)
  self.state_pub=self.create_publisher(Float64MultiArray,f'/swarm/robot_{self.robot_id}/state',10)
  self.create_subscription(Odometry,'odom',self.odom,10)
  self.subscriptions_peers=[self.create_subscription(Float64MultiArray,f'/swarm/robot_{i}/state',self.peer,10) for i in range(self.n) if i!=self.robot_id]
  self.timer=self.create_timer(.1,self.tick)
  self.get_logger().info('Shadow output only: policy/cmd_vel; awaiting common-frame odometry.')

 def odom(self,msg):
  q=msg.pose.pose.orientation
  yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
  self.state=(msg.pose.pose.position.x,msg.pose.pose.position.y,yaw,msg.twist.twist.linear.x,msg.twist.twist.angular.z,time.monotonic())

 def peer(self,msg):
  data=list(msg.data)
  if len(data)!=8 or not all(math.isfinite(x) for x in data):return
  sender=int(data[0])
  if sender==self.robot_id or not 0<=sender<self.n:return
  previous=self.received.get(sender)
  if previous is None or data[1]>previous[1]:self.received[sender]=data

 def tick(self):
  output=Twist();now=self.get_clock().now().nanoseconds/1e9
  if self.state is None or time.monotonic()-self.state[5]>self.max_age:
   self.last_command=(0.,0.);self.publisher.publish(output);return
  x,y,heading,v,w,_=self.state
  message=Float64MultiArray();message.data=[float(self.robot_id),float(self.seq),now,x,y,heading,v,w]
  self.seq+=1;self.state_pub.publish(message)
  limits=self.limits
  local=[v,w]+[limits[k] for k in ['max_forward_speed_mps','max_reverse_speed_mps','max_yaw_rate_radps','max_linear_acceleration_mps2','max_angular_acceleration_radps2','footprint_radius_m']]+[self.task['team']['desired_pair_distance_m']]
  rows=[];masks=[];c=math.cos(heading);s=math.sin(heading)
  for i in range(self.n):
   if i==self.robot_id:continue
   peer=self.received.get(i);age=now-peer[2] if peer else -1.
   if peer is None or not 0<=age<=self.max_age:
    rows.extend([0.]*7);masks.append(0.);continue
   dx,dy=peer[3]-x,peer[4]-y;angle=peer[5]-heading
   rows.extend([c*dx+s*dy,-s*dx+c*dy,math.sin(angle),math.cos(angle),peer[6],peer[7],age/self.max_age]);masks.append(1.)
  try:
   if self.process is not None:
    self.process.stdin.write(json.dumps({'observation':local+rows+masks})+'\n');self.process.stdin.flush()
    if not select.select([self.process.stdout],[],[],.5)[0]:raise TimeoutError('Actor inference timeout')
    action=json.loads(self.process.stdout.readline())['action']
    if len(action)!=2 or not all(math.isfinite(a) for a in action):raise ValueError('Invalid actor action')
    a,b=[max(-1.,min(1.,a)) for a in action]
    target_v=a*(limits['max_forward_speed_mps'] if a>=0 else limits['max_reverse_speed_mps'])
    target_w=b*limits['max_yaw_rate_radps']
    pv,pw=self.last_command;dv=limits['max_linear_acceleration_mps2']*.1;dw=limits['max_angular_acceleration_radps2']*.1
    output.linear.x=pv+max(-dv,min(dv,target_v-pv));output.angular.z=pw+max(-dw,min(dw,target_w-pw))
  except Exception as error:
   self.get_logger().error(str(error))
   if self.process is not None:self.process.terminate();self.process=None
  self.last_command=(output.linear.x,output.angular.z)
  self.publisher.publish(output)

 def destroy_node(self):
  self.publisher.publish(Twist())
  if self.process is not None:self.process.terminate();self.process.wait(timeout=5)
  return super().destroy_node()

def main(args=None):
 rclpy.init(args=args);node=PolicyNode()
 try:rclpy.spin(node)
 except KeyboardInterrupt:pass
 finally:node.destroy_node();rclpy.shutdown()
