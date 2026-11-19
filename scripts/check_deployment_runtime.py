"""Check the local runtime boundary and message/session/watchdog failures."""
import copy,json,math
import numpy as np
from swarm_core.deployment.local_runtime import LocalMissionRuntime
from swarm_core.deployment.protocol import PeerEnvelope
from swarm_core.communication.message import StateMessage
from swarm_core.dynamics import RobotState
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.mappo_adapter import pack

task=json.load(open('configs/tasks/history_full_mission.json'))
class Policy:
    def act_local(self,o,robot):return np.array([.2,.1]),'fixture_policy'
def runtime():return LocalMissionRuntime(task,0,'mission_test','boot_0','mission',Policy())
def packet(sender=1,sequence=0,stamp=0.,session='boot_1',mission='mission_test',frame='mission',fault=False):
    state=StateMessage(sender_id=sender,sequence=sequence,generated_at_s=stamp,x=1. if sender==1 else .5,y=0. if sender==1 else math.sqrt(3)/2,heading=0.,speed=0.,yaw_rate=0.,payload_size_bytes=72)
    return PeerEnvelope(mission,session,frame,state,fault).to_dict()
r=runtime();assert r.tick(RobotState(),0.,0.)['mode']=='awaiting_peers'
assert not r.accept_peer(packet(mission='other'),.1) and not r.peer_sessions
assert not r.accept_peer(packet(frame='other'),.1) and not r.peer_sessions
assert not r.accept_peer(packet(stamp=1.),.1)
assert r.accept_peer(packet(),.1)
assert not r.accept_peer(packet(),.1)
assert not r.accept_peer(packet(sequence=1,session='boot_new'),.2) and r.fault=='peer_session_changed'
assert r.tick(RobotState(),.2,.2)['command']==[0.,0.]
r=runtime();r.tick(RobotState(),0.,0.)
assert r.tick(RobotState(),.4,0.)['fault']=='invalid_or_stale_odometry'
assert r.tick(RobotState(),.5,.5)['command']==[0.,0.], 'Fresh data cannot silently clear a latched fault.'
r=runtime();assert r.accept_peer(packet(fault=True),0.) and r.fault=='peer_faulted'
r=runtime();r.tick(RobotState(),1.,1.);assert r.tick(RobotState(),.9,.9)['command']==[0.,0.]
# Observation construction must match the simulation without remote truth.
env=MissionEnv(task,json.load(open('configs/communication/nominal.json')),{});observations,_=env.reset(42)
for robot in range(3):
    r=LocalMissionRuntime(task,robot,'mission_test','own_'+str(robot),'mission',Policy())
    for sender,message in env.simulation.network.receivers[robot]._latest.items():
        assert r.accept_peer(PeerEnvelope('mission_test','peer_'+str(sender),'mission',message).to_dict(),0.)
    own=env.simulation.estimates[robot];actual=r.observation(own,0.)
    np.testing.assert_array_equal(pack([actual]),pack([observations[robot]]))
    np.testing.assert_array_equal(actual['formation_history']['neighbor_features'],observations[robot]['formation_history']['neighbor_features'])
    r.phase=1;r.goal=np.array([own.x+5.,own.y]);env.phases[robot]=1;env.goals[robot]=r.goal.copy()
    np.testing.assert_array_equal(pack([r.observation(own,0.)]),pack([env.observe_all()[robot]]))
print('Mission/session/frame validation, duplicate rejection, fault propagation and watchdogs: PASSED')
print('Local versus simulator observation construction: PASSED')
