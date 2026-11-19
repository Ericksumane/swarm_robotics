"""Check live models have physical contacts, actuators and unique feedback."""
import xml.etree.ElementTree as ET
from swarm_core.gazebo_world import build_live_world
states=[dict(x=float(i),y=0.,heading=0.) for i in range(3)]
world=ET.fromstring(build_live_world(states)).find('world');assert world.attrib['name']=='swarm_live'
for i in range(3):
 model=world.find(f"model[@name='robot_{i}']");assert model.findtext('static')!='true'
 assert len(model.findall('.//collision'))>=3 and len(model.findall("joint[@type='revolute']"))==2
 drive=model.find("plugin[@name='gz::sim::systems::DiffDrive']");odom=model.find("plugin[@name='gz::sim::systems::OdometryPublisher']")
 assert drive.findtext('topic')==f'/model/robot_{i}/cmd_vel'
 assert odom.findtext('odom_topic')==f'/model/robot_{i}/odometry'
 assert odom.findtext('odom_frame')=='world'
print('Dynamic robot collisions, wheel actuation and world-frame feedback: PASSED')
