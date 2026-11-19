from glob import glob
from setuptools import setup
setup(name='swarm_runtime',version='0.1.0',packages=['swarm_runtime'],
 data_files=[('share/ament_index/resource_index/packages',['resource/swarm_runtime']),
 ('share/swarm_runtime',['package.xml']),('share/swarm_runtime/launch',glob('launch/*.launch.py'))],
 entry_points={'console_scripts':['policy_node = swarm_runtime.policy_node:main','full_mission_node = swarm_runtime.full_mission_node:main']})
