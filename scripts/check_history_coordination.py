"""Exercise lossy local readiness, proposal negotiation and revoked evidence."""
import math,json
import numpy as np
from swarm_core.communication.network import Network
from swarm_core.communication.message import StateMessage
from swarm_core.communication.history import build_history_observation
from swarm_core.history_startup import EvidenceWindow,HistoryStartupCoordinator
from swarm_core.dynamics import RobotState

settings=dict(required_evidence_s=2.,max_evidence_gap_s=2.,certificate_ttl_s=3.,proposal_lead_s=6.,slot_width_s=5.,geometry_tolerance_m=.12,heading_tolerance_rad=.08)
history_settings=json.load(open('configs/tasks/dispersed_history_training.json'))['peer_history']
e=EvidenceWindow()
for i in range(25):assert not e.update(True,False,[1,1],i*.1), 'One stale packet cannot certify repeated evidence.'
assert e.update(True,False,[2,2],2.5)
assert not e.update(False,True,[2,2],2.6)
states=[RobotState(0.,0.),RobotState(1.,0.),RobotState(.5,math.sqrt(3)/2)]
results={}
for profile in ['nominal','severe']:
    comm=json.load(open('configs/communication/'+profile+'.json'))
    network=Network(3,delay_s=comm['delay_s'],loss_probability=comm['loss_probability'],max_age_s=comm['max_information_age_s'],seed=42,channel_options=comm.get('channel_options',{}))
    nodes=[HistoryStartupCoordinator(settings) for _ in states];ready=[False]*3;proposals=[None]*3;starts=[None]*3
    for tick in range(900):
        now=tick*.1
        for i,s in enumerate(states):
            network.broadcast(StateMessage(sender_id=i,sequence=tick,generated_at_s=now,**vars(s),payload_size_bytes=72,formation_ready=ready[i],start_at_s=proposals[i],start_committed=nodes[i].committed),now)
        network.deliver(now)
        for i,node in enumerate(nodes):
            if starts[i] is not None:continue
            h=build_history_observation(states[i],network.receivers[i],[j for j in range(3) if j!=i],now,1.,history_settings)
            ready[i],proposals[i],start=node.update(states[i],network.receivers[i],h,now,[1.,0.])
            if start:starts[i]=now
        if all(s is not None for s in starts):break
    assert all(s is not None for s in starts),(profile,starts,proposals)
    assert max(starts)-min(starts)<.11,(profile,starts)
    results[profile]=starts
# No messages, even with perfect own pose, cannot authorize departure.
network=Network(3);node=HistoryStartupCoordinator(settings)
for tick in range(50):
    h=build_history_observation(states[0],network.receivers[0],[1,2],tick*.1,1.,history_settings)
    ready,proposal,start=node.update(states[0],network.receivers[0],h,tick*.1,[1.,0.])
    assert not ready and proposal is None and not start
print('No fabricated evidence, explicit expiry, leaderless future-slot negotiation: PASSED')
print(results)

# An unoffered or expired gossiped commitment cannot authorize departure.
from swarm_core.communication.receiver import Receiver
receiver=Receiver(0,.5)
node=HistoryStartupCoordinator(settings)
msg=StateMessage(sender_id=1,sequence=1000,generated_at_s=0.,x=1.,y=0.,heading=0.,speed=0.,yaw_rate=0.,payload_size_bytes=72,formation_ready=True,start_at_s=10.,start_committed=True)
receiver.accept(msg,5.)
h=build_history_observation(states[0],receiver,[1,2],5.,1.,history_settings)
assert not node.update(states[0],receiver,h,5.,[1.,0.])[2] and not node.committed
# Own displacement invalidates a previously accepted promise to hold position.
node.committed=True;node.certificate=True;node.proposal=10.;node.offered_positions[10.]=np.array([0.,0.])
ready,proposal,start=node.update(RobotState(x=.1),receiver,h,5.,[1.,0.])
assert not ready and not start and not node.committed
print('Expired/unoffered commitments and own-displacement revocation: PASSED')


# Check the configured triangle size through mission reset and the actual
# nominal message exchange. A hard-coded 1 m readiness gate fails this case.
from swarm_core.mission_env import MissionEnv

task = json.load(open('configs/tasks/history_full_mission.json'))
task['team']['desired_pair_distance_m'] = 1.2
env = MissionEnv(task, json.load(open('configs/communication/nominal.json')), {})
env.reset(42)
env.simulation.states = [
    RobotState(0., 0.), RobotState(1.2, 0.),
    RobotState(.6, .6 * math.sqrt(3.)),
]
env.simulation._measure()
env.simulation.reset_network(42)
for _ in range(300):
    env.simulation.advance(np.zeros((3, 2)))
    env._update_phases()
    if np.all(env.phases == 1):
        break
assert np.all(env.phases == 1), 'The 1.2 m triangle failed startup.'
assert all(abs(goal[0] - start[0] - 5.) < 1e-9
           for start, goal in zip(env.starts, env.goals))
print('Configured 1.2 m formation advances to MOVE: PASSED')
