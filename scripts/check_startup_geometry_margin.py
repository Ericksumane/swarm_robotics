"""Check stricter startup geometry without relaxing physical success criteria."""
import json,math
from dataclasses import replace
import numpy as np
from swarm_core.mission_env import MissionEnv
task=json.load(open('configs/tasks/dispersed_full_tight_start.json'));comm=json.load(open('configs/communication/nominal.json'));env=MissionEnv(task,comm,{})
assert task['mission']['formation_tolerance_m']==.1 and task['mission']['destination_tolerance_m']==.2
# Analytically known test fixture; evaluation rollouts never edit poses.
env.reset(1);side=1.07;env.simulation.states=[replace(s,x=x,y=y,heading=0.,speed=0.,yaw_rate=0.) for s,(x,y) in zip(env.simulation.states,[(0.,0.),(side,0.),(side/2,side*math.sqrt(3)/2)])];env.simulation._measure();env.simulation.reset_network(1)
for _ in range(25):env.step(np.zeros((3,2)))
assert np.all(env.phases==0) and not any(env.simulation.formation_ready)
print('Strict startup margin preserves final geometry and goal criteria: PASSED')
