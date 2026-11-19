"""Check optional heading readiness without altering legacy transitions."""
import json,copy,math
from dataclasses import replace
import numpy as np
from swarm_core.mission_env import MissionEnv
from swarm_core.learning.acquisition_router import LocalAcquisitionRouter
task=json.load(open('configs/tasks/dispersed_full_aligned.json'));comm=json.load(open('configs/communication/nominal.json'))
env=MissionEnv(task,comm,{})
# Known-state fixtures only; no pose changes are used in performance evaluations.
env.reset(1);env.simulation.states=[replace(s,x=x,y=y,heading=math.pi/2,speed=0.,yaw_rate=0.) for s,(x,y) in zip(env.simulation.states,[(0.,0.),(1.,0.),(.5,math.sqrt(3)/2)])];env.simulation._measure();env.simulation.reset_network(1)
for _ in range(25):env.step(np.zeros((3,2)))
assert np.all(env.phases==0) and not any(env.simulation.formation_ready)
env.simulation.states=[replace(s,heading=0.) for s in env.simulation.states];env.simulation._measure()
for _ in range(25):env.step(np.zeros((3,2)))
assert np.all(env.phases==1)
env.reset(1);env.simulation.states=[replace(s,x=x,y=y,heading=math.pi/2,speed=0.,yaw_rate=0.) for s,(x,y) in zip(env.simulation.states,[(0.,0.),(1.,0.),(.5,math.sqrt(3)/2)])];env.simulation._measure();env.simulation.reset_network(1)
o=env.observe_all()[0];router=LocalAcquisitionRouter()
for _ in range(19):assert router.select(o)=='formation'
assert router.select(o)=='alignment'
missing=copy.deepcopy(o);missing['neighbor_mask'][0]=0;assert router.select(missing)=='formation'
assert not router.alignment_latched
print('Heading readiness, observed-geometry alignment routing and reset: PASSED')
