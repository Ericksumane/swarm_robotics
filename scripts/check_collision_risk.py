"""Check independent closest-approach cases and reward accounting."""
import math,json
import numpy as np
from swarm_core.collision_risk import closing_risk
from swarm_core.dynamics import RobotState
from swarm_core.formation_env import FormationEnv

approaching=[RobotState(-.5,0.,0.,.2),RobotState(.5,0.,math.pi,.2)]
departing=[RobotState(-.5,0.,math.pi,.2),RobotState(.5,0.,0.,.2)]
assert closing_risk(approaching,.25,2.,.3)>1
assert closing_risk(departing,.25,2.,.3)==0
stationary=[RobotState(-.5),RobotState(.5)]
assert closing_risk(stationary,.25,2.,.3)==0
task=json.load(open('configs/tasks/history_collision_training.json'));comm=json.load(open('configs/communication/nominal.json'));weights=json.load(open('configs/training/history_collision_reward.json'))
env=FormationEnv(task,comm,weights);env.reset(83000003)
obs,rewards,*rest=env.step(np.zeros((3,2)))
assert np.isfinite(rewards).all()
assert np.isclose(rewards[0],sum(rest[-1]['reward_terms'].values()))
assert rest[-1]['reward_terms']['closing_risk']<=0
print('Approaching versus receding risk and reward accounting: PASSED')
