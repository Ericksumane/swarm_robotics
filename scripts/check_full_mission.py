"""Contract tests with explicit state fixtures, NOT a learned demonstration."""
import json,math
import numpy as np
from swarm_core.mission_env import MissionEnv,FORM,MOVE,HOLD
from swarm_core.dynamics import RobotState
def main():
 task=json.load(open('configs/tasks/full_mission.json'));comm=json.load(open('configs/communication/nominal.json'))
 env=MissionEnv(task,comm,{})
 # Test fixture: exact formation, stationary, aligned with mission direction.
 initial=[RobotState(x=0,y=0),RobotState(x=1,y=0),RobotState(x=.5,y=math.sqrt(3)/2)]
 env.simulation.states=initial;env.simulation._measure();env.simulation._transmit();env.simulation.network.deliver(0.)
 for _ in range(25):env.step(np.zeros((3,2)))
 assert np.all(env.phases==MOVE)
 for start,goal in zip(env.starts,env.goals):np.testing.assert_allclose(goal-start,[5.,0.])
 goals=[g.copy() for g in env.goals]
 # Explicit endpoint fixture checks success semantics; it does not test tracking.
 env.simulation.states=[RobotState(x=float(g[0]),y=float(g[1])) for g in goals]
 for _ in range(25):
  _,_,term,trunc,info=env.step(np.zeros((3,2)))
  if term:break
 assert term and not trunc and info['reason']=='success'
 assert abs(info['translation_m']-5.)<1e-8
 assert np.all(env.phases==HOLD)
 assert env.observe_all()[0]['local_features'].shape==(16,)
 blocked=dict(comm,loss_probability=1.)
 isolated=MissionEnv(task,blocked,{})
 isolated.simulation.states=initial;isolated.simulation._measure()
 for _ in range(25):isolated.step(np.zeros((3,2)))
 assert np.all(isolated.phases==FORM)
 # Exercise actual motion integration with explicit test commands.
 # These commands are a contract test, not a learned or deployed policy.
 moving=MissionEnv(task,comm,{})
 moving.simulation.states=initial;moving.simulation._measure();moving.simulation._transmit();moving.simulation.network.deliver(0.)
 for _ in range(25):moving.step(np.zeros((3,2)))
 for _ in range(800):
  action=np.zeros((3,2))
  if not np.all(moving.phases==HOLD):action[:,0]=.5
  _,_,term,trunc,info=moving.step(action)
  if term or trunc:break
 assert info['reason']=='success'
 assert abs(info['translation_m']-5.)<=task['mission']['destination_tolerance_m']
 assert all(abs(s.speed)<=task['mission']['stop_linear_speed_tolerance_mps'] for s in moving.simulation.states)
 print('Integrated translation and stop test: PASSED (explicit test actions)')
 print('Local readiness, fixed 5 m goals, final stop and loss-blocked start: PASSED')
 print('State fixtures validate mission logic only, not policy competence.')
if __name__=='__main__':main()
