"""Validate reset difficulty, mastery promotion and success-aligned rewards."""
import json, math
from copy import deepcopy
from dataclasses import replace
import numpy as np
from swarm_core.curriculum import HeadingCurriculum, sample_headings
from swarm_core.initialization import sample_initial_states
from swarm_core.formation_env import FormationEnv
from swarm_core.geometry import formation_metrics
from swarm_core.dynamics import RobotState

def main():
    load=lambda name: json.load(open(name))
    task=load('configs/tasks/formation_mastery.json')
    curriculum=HeadingCurriculum(task['training_curriculum'])
    for _ in range(19): assert not curriculum.record(True)
    assert curriculum.record(True) and curriculum.stage==1
    for _ in range(20): assert not curriculum.record(False)
    assert curriculum.stage==1
    assert len(curriculum.history)==20
    # An aligned reset must still start outside the unchanged success tolerance.
    for seed in range(30):
        states=sample_initial_states(task,seed)
        other=sample_initial_states(task,seed)
        assert states==other
        before=formation_metrics(states,1.,.25)
        assert before['max_pair_error_m']>=.12
        aligned=deepcopy(task)
        aligned['initialization']['heading_alignment']['half_width_rad']=0.
        states=sample_initial_states(aligned,seed)
        moved=[replace(s,x=s.x+1e-5*math.cos(s.heading),y=s.y+1e-5*math.sin(s.heading)) for s in states]
        assert formation_metrics(moved,1.,.25)['formation_rmse_m'] < formation_metrics(states,1.,.25)['formation_rmse_m']
    try: sample_headings([[0,0],[1,0],[.5,.8]],1.,{'half_width_rad':float('nan')},np.random.default_rng(0))
    except ValueError: pass
    else: raise AssertionError('Invalid alignment accepted.')
    bad=deepcopy(task);bad['initialization']['near_formation']['probability']=.5
    try: sample_initial_states(bad,42)
    except ValueError: pass
    else: raise AssertionError('Mixed reset alignment silently ignored.')
    comm=load('configs/communication/nominal.json')
    weights=load('configs/training/formation_reward_mastery.json')
    env=FormationEnv(task,comm,weights);env.reset(42)
    env.simulation.states=[RobotState(x=0,y=0),RobotState(x=1,y=0),RobotState(x=.5,y=math.sqrt(3)/2)]
    for i in range(env.monitor.required_steps):
        _,reward,term,trunc,info=env.step(np.zeros((3,2)))
        assert math.isclose(info['reward_terms']['ready'],.05)
        assert term == (i==env.monitor.required_steps-1)
        np.testing.assert_allclose(reward,sum(info['reward_terms'].values()),atol=1e-6)
    assert info['reason']=='success' and not trunc
    assert info['reward_terms']['success']==2.
    # Verify max-pair shaping independently from the RMSE implementation.
    env.reset(43)
    previous=formation_metrics(env.simulation.states,1.,.25)['max_pair_error_m']
    _,_,_,_,info=env.step(np.full((3,2),.1))
    current=formation_metrics(env.simulation.states,1.,.25)['max_pair_error_m']
    assert math.isclose(info['reward_terms']['progress'],20*(previous-current),abs_tol=1e-10)
    assert info['reward_terms']['ready']==0.
    print('Mastery promotion and failure isolation: PASSED')
    print('Nontrivial reproducible initialization and heading descent: PASSED')
    print('Dense hold credit and unchanged two-second success: PASSED')
    print('Worst-pair progress accounting: PASSED')

if __name__=='__main__': main()
