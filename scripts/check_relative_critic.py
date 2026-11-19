"""Relative critic representation must be invariant to global rigid transforms."""
import json,math
from dataclasses import replace
import numpy as np
from swarm_core.formation_env import FormationEnv
from swarm_core.learning.mappo_adapter import critic_input
def main():
 task=json.load(open('configs/tasks/formation_relative_critic.json'));comm=json.load(open('configs/communication/nominal.json'));weights=json.load(open('configs/training/formation_reward_proximity.json'))
 env=FormationEnv(task,comm,weights)
 before=critic_input(env);assert before.shape==(3,23)
 angle=.73;c,s=math.cos(angle),math.sin(angle)
 env.simulation.states=[replace(p,x=c*p.x-s*p.y+12.,y=s*p.x+c*p.y-7.,heading=p.heading+angle) for p in env.simulation.states]
 np.testing.assert_allclose(before,critic_input(env),atol=1e-6)
 print('Relative critic width and global translation/rotation invariance: PASSED')
if __name__=='__main__':main()
