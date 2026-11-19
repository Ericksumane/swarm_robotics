"""Check physical prediction, zero-age identity and message masking."""
from dataclasses import replace
import math
import numpy as np
from swarm_core.communication.message import StateMessage
from swarm_core.communication.receiver import Receiver
from swarm_core.communication.prediction import predict_received_pose
from swarm_core.dynamics import RobotState,step
from swarm_core.observations import build_neighbor_observation

def main():
 limits=dict(max_forward_speed_mps=.3,max_reverse_speed_mps=.15,max_yaw_rate_radps=.8,max_linear_acceleration_mps2=.3,max_angular_acceleration_radps2=.8)
 for v,w,heading in [(.2,0.,.3),(.2,.4,-.8),(-.1,-.4,1.1)]:
  state=RobotState(x=1.,y=-.3,heading=heading,speed=v,yaw_rate=w)
  message=StateMessage(sender_id=1,sequence=0,generated_at_s=0.,x=state.x,y=state.y,heading=heading,speed=v,yaw_rate=w,payload_size_bytes=64)
  target=step(state,[v/(.3 if v>=0 else .15),w/.8],limits,.37)
  np.testing.assert_allclose(predict_received_pose(message,.37),[target.x,target.y,target.heading],atol=1e-12)
  assert predict_received_pose(message,0.)==(message.x,message.y,message.heading)
 receiver=Receiver(0,.5);message=StateMessage(sender_id=1,sequence=0,generated_at_s=0.,x=1.,y=0.,heading=0.,speed=.1,yaw_rate=0.,payload_size_bytes=64);receiver.accept(message,0.)
 own=RobotState(x=.01,speed=.1)
 legacy=build_neighbor_observation(own,receiver,[1],.1)
 predicted=build_neighbor_observation(own,receiver,[1],.1,predict_to_now=True)
 assert np.isclose(legacy['neighbor_features'][0,0],.99)
 assert np.isclose(predicted['neighbor_features'][0,0],1.)
 assert predicted['neighbor_features'][0,6]==legacy['neighbor_features'][0,6]
 expired=build_neighbor_observation(own,receiver,[1],.6,predict_to_now=True)
 assert not expired['neighbor_mask'].any() and not expired['neighbor_features'].any()
 try:predict_received_pose(message,-.1)
 except ValueError:pass
 else:raise AssertionError('Negative age accepted')
 print('Straight, turning and reverse payload-only prediction: PASSED')
 print('Original age, zero-age identity and expired-message masking: PASSED')

if __name__=='__main__':main()
