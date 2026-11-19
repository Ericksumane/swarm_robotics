"""Check alignment success, non-success headings, reset and critic interface."""
import json
from dataclasses import replace
import numpy as np
from swarm_core.alignment_env import AlignmentEnv
from swarm_core.learning.mappo_adapter import critic_input,pack
task=json.load(open('configs/tasks/alignment_v1.json'));comm=json.load(open('configs/communication/nominal.json'));weights=json.load(open('configs/training/alignment_reward.json'))
env=AlignmentEnv(task,comm,weights);obs,_=env.reset(7)
assert pack(obs).shape==(3,36) and critic_input(env).shape==(3,39)
assert all(o['goal_distance_m'] is None for o in obs) and all(env.phases==0)
assert max(env.heading_errors())>.04
for _ in range(20):_,_,term,trunc,info=env.step(np.zeros((3,2)))
assert not term and not trunc
env.reset(7);env.simulation.states=[replace(s,heading=0.,speed=0.,yaw_rate=0.) for s in env.simulation.states];env.simulation._measure()
for _ in range(20):_,_,term,trunc,info=env.step(np.zeros((3,2)))
assert term and not trunc and info['reason']=='success'
try:env.step(np.zeros((3,2)))
except RuntimeError:pass
else:raise AssertionError('Post-end step accepted.')
env.reset(7);assert not env.done and env.success_steps==0
print('Alignment criteria, finite actor/critic inputs and reset: PASSED')
