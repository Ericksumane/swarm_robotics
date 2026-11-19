"""Check supervisor-matched parked peers, goal shaping and training-only failure."""
from copy import deepcopy
import json,math
import numpy as np
from swarm_core.arrival_training import ArrivalTrainingEnv

from swarm_core.motion_profile import scaled_linear_speed
task=scaled_linear_speed(json.load(open('configs/tasks/translate_stop_hold_hysteresis.json')),1.1)
task['arrival_training']=dict(parking=dict(entry_m=.19,exit_m=.2,settle_s=2.,time_step_s=.1,linear_tolerance_mps=.03,yaw_tolerance_radps=.05),motion_gate=None,inner_target_m=.16,distance_scale_m=.2,excess_weight=2.,excess_progress_weight=20.,exit_event_penalty=5.,deadlock_penalty=40.,timeout_penalty=20.)
reward=json.load(open('configs/training/approach_reward.json'));comm=json.load(open('configs/communication/nominal.json'))
goals=np.array([[5.,0.],[6.,0.],[5.5,math.sqrt(3)/2]])
def fixture(offsets,parked):
 t=deepcopy(task);states=[dict(x=float(g[0]+d),y=float(g[1]),heading=0.,speed=0.,yaw_rate=0.) for g,d in zip(goals,offsets)]
 t['approach_reset_bank']=dict(entries=[dict(states=states,starts=(goals-[5.,0.]).tolist(),goals=goals.tolist(),phases=[1 if not p else 2 for p in parked],parked=parked)])
 return ArrivalTrainingEnv(t,comm,reward)
# Parked peers stay stationary even if the actor emits nonzero raw actions.
forward=fixture([-.24,-.1,-.1],[False,True,True]);reverse=fixture([-.24,-.1,-.1],[False,True,True]);forward.reset(3);reverse.reset(3);before=forward.simulation.states.copy()
_,_,_,_,f=forward.step(np.ones((3,2))*.5);_,_,_,_,r=reverse.step(-np.ones((3,2))*.5)
assert forward.simulation.states[1:]==before[1:]
np.testing.assert_array_equal(forward.last_actor_active_masks[:,0],[1,0,0])
assert f['reward_terms']['arrival_excess_progress']>0>r['reward_terms']['arrival_excess_progress']
# A correct parked formation must be allowed to accumulate the unchanged hold.
valid=fixture([-.1,-.1,-.1],[True]*3);valid.reset(4)
for _ in range(20):_,_,term,trunc,info=valid.step(np.ones((3,2)))
assert term and not trunc and info['reason']=='success'
# An unrecoverable fully parked geometry is an explicit TRAINING failure.
dead=fixture([-.14,.14,0.],[True]*3);dead.reset(5);_,_,term,trunc,info=dead.step(np.ones((3,2)))
assert term and not trunc and info['reason']=='training_park_deadlock'
assert info['reward_terms']['arrival_park_deadlock']<0
try:dead.step(np.zeros((3,2)))
except RuntimeError:pass
else:raise AssertionError('Post-end training step accepted')
print('Parked-peer isolation, directed goal shaping, real hold and training deadlock: PASSED')
