"""The optional motion-cost gate must affect acquisition only."""
import json,math
from copy import deepcopy
from dataclasses import replace
import numpy as np
from swarm_core.formation_env import FormationEnv
from swarm_core.dynamics import RobotState

def main():
    load=lambda p:json.load(open(p))
    task=load('configs/tasks/formation_relational_random_evaluation.json')
    comm=load('configs/communication/nominal.json')
    original=load('configs/training/formation_reward_mastery.json')
    revised=load('configs/training/formation_reward_acquisition.json')
    triangle=[RobotState(x=0,speed=.1),RobotState(x=1,speed=.1),RobotState(x=.5,y=math.sqrt(3)/2,speed=.1)]
    for scale in (1.,1.3):
        fixture=[replace(s,x=scale*s.x,y=scale*s.y) for s in triangle]
        records=[]
        for weights in (original,revised):
            env=FormationEnv(task,comm,weights);env.reset(42);env.simulation.states=fixture
            _,r,term,trunc,info=env.step(np.zeros((3,2)))
            np.testing.assert_allclose(r,sum(info['reward_terms'].values()),atol=1e-6)
            records.append((r,info))
        old,new=records
        assert old[1]['reward_terms']['motion']<0
        if scale==1.:
            assert new[1]['diagnostics']['geometry_ready']
            np.testing.assert_allclose(new[0],old[0])
        else:
            assert not new[1]['diagnostics']['geometry_ready']
            assert new[1]['reward_terms']['motion']==0.
            np.testing.assert_allclose(new[0]-old[0],-old[1]['reward_terms']['motion'],atol=1e-6)
        assert old[1]['diagnostics']==new[1]['diagnostics']
    print('Acquisition-only motion-cost gate, legacy reward and accounting: PASSED')

if __name__=='__main__':main()
