"""Check phase hysteresis while keeping physical success criteria unchanged."""
from dataclasses import replace
import json
import numpy as np
from swarm_core.mission_env import MissionEnv,MOVE,HOLD

def pose(env,distance):
 env.simulation.states=[replace(s,x=float(g[0]-distance),y=float(g[1]),heading=0.,speed=0.,yaw_rate=0.) for s,g in zip(env.simulation.states,env.goals)]
 env.simulation._measure()

def main():
 task=json.load(open('configs/tasks/translate_stop_hold_hysteresis.json'));comm=json.load(open('configs/communication/nominal.json'));env=MissionEnv(task,comm,{})
 pose(env,.18);env._update_phases();assert np.all(env.phases==MOVE)
 pose(env,.14);env._update_phases();assert np.all(env.phases==HOLD)
 pose(env,.18);env._update_phases();assert np.all(env.phases==HOLD)
 pose(env,.21);env._update_phases();assert np.all(env.phases==MOVE)
 # Explicit stationary fixture, not a policy demonstration. Physical completion
 # must not depend on an arbitrary internal phase label.
 env.reset(91);pose(env,.18)
 for _ in range(20):
  _,_,term,trunc,info=env.step(np.zeros((3,2)))
 assert term and not trunc and info['reason']=='success' and np.all(env.phases==MOVE)
 invalid=json.loads(json.dumps(task));invalid['mission']['hold_entry_tolerance_m']=.3
 try:MissionEnv(invalid,comm,{})
 except ValueError:pass
 else:raise AssertionError('Invalid hysteresis accepted')
 print('Inner HOLD entry, original exit and history retention: PASSED')
 print('Unchanged physical goal/stop/formation success test: PASSED')

if __name__=='__main__':main()
