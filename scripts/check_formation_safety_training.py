"""Behavior checks for training-only formation clearance shaping."""
from copy import deepcopy
import json,math
import numpy as np
from swarm_core.dynamics import RobotState
from swarm_core.formation_safety_training import FormationSafetyTrainingEnv
from swarm_core.motion_profile import scaled_linear_speed
from swarm_core.learning.mappo_adapter import pack

task=scaled_linear_speed(json.load(open('configs/tasks/dispersed_history_training.json')),1.1)
task['formation_safety_training']=dict(target_clearance_m=.2,near_weight=60.,recovery_progress_weight=8.,motion_gate=dict(horizon_s=1.,prediction_step_s=.1,clearance_margin_m=.03,history_bound_scale_m=1.,veto_when_missing_form=True,veto_when_missing=False,uncertainty_in_move=False))
weights=json.load(open('configs/training/history_collision_reward.json'));comm=json.load(open('configs/communication/nominal.json'))
def fixture(positions):
 env=FormationSafetyTrainingEnv(task,comm,weights);env.reset(7);env.simulation.states=[RobotState(x=x,y=y) for x,y in positions];env.simulation._measure();env.simulation.reset_network(7);return env
far=fixture([(0,0),(1,0),(.5,math.sqrt(3)/2)]);near=fixture([(0,0),(.6,0),(.3,1.2)])
obs,_,_,_,f=far.step(np.zeros((3,2)));_,_,_,_,n=near.step(np.zeros((3,2)))
assert pack(obs).shape==(3,33)
assert f['reward_terms']['near_clearance']==0.
assert n['reward_terms']['near_clearance']<0.
assert math.isclose(n['minimum_clearance_m'],.1)
# Same operational gate: unknown peer estimates veto translation in FORM.
loss=json.load(open('configs/communication/nominal.json'));loss['loss_probability']=1.
blocked=FormationSafetyTrainingEnv(task,loss,weights);blocked.reset(8);before=blocked.simulation.states.copy();blocked.step(np.tile([1.,0.],(3,1)))
assert blocked.simulation.states==before
assert task['formation_safety_training']['motion_gate']['clearance_margin_m']==.03
print('Localized clearance cost, original actor contract and unchanged FORM veto: PASSED')
