"""Local stop evidence, position hold, time gaps and disturbance release."""
import numpy as np
from swarm_core.dynamics import RobotState
from swarm_core.goal_stop import GoalStopLatch, limit_forward_for_arrival
settings=dict(entry_m=.19,exit_m=.2,settle_s=2.,time_step_s=.1,linear_tolerance_mps=.03,yaw_tolerance_radps=.05)
stop=GoalStopLatch(settings);a=np.array([.05,.01]);local=np.zeros(16);local[10]=1.
def observation(t,d=.18,v=0.):
    values=local.copy();values[0]=v
    return dict(control_time_s=t,goal_distance_m=d,local_features=values)
for i in range(19):assert not stop.apply(a,observation(i*.1))[1]
result,parked=stop.apply(a,observation(1.9));assert parked and not result.any()
assert stop.apply(a,observation(30.))[1]
assert not stop.apply(a,observation(30.1,.201))[1]
stop.reset()
for i in range(19):stop.apply(a,observation(i*.1))
assert not stop.apply(a,observation(10.))[1], 'A long inference gap cannot count as settled time.'
stop.reset()
for i in range(30):assert not stop.apply(a,observation(i*.1,.18,.04))[1]
stop.reset();o=observation(0.);o['goal_distance_m']=None;o['local_features'][9:12]=[1.,0.,0.]
assert not stop.apply(a,o)[1]
print('Local stop evidence, no drifting action, time gaps and disturbance release: PASSED')

limits = {"max_forward_speed_mps": 1.6}
own = RobotState(speed=.8)
command = np.array([1., .2])
goal = np.array([.4, 0.])
capped = limit_forward_for_arrival(command, own, goal, limits, .1, .3)
np.testing.assert_allclose(capped, [np.sqrt(2 * .3 * .32) / 1.6, .2])
np.testing.assert_array_equal(command, [1., .2])
assert limit_forward_for_arrival(command, own, -goal, limits, .1, .3)[0] == 0.
assert limit_forward_for_arrival(command, own, goal, limits, .1, None) is command
print("Speed-dependent forward braking envelope: PASSED")
