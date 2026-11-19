"""Hard reset variation must preserve physical constraints and reproducibility."""
import json,math
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from swarm_core.initialization import sample_initial_states
from swarm_core.geometry import formation_metrics

def main():
    task=json.load(open('configs/tasks/formation_relational_120s_training.json'))
    initial=sample_initial_states(task,12345)
    original=deepcopy(task)
    task['initialization']['hard_examples']=dict(probability=1.,position_jitter_m=.04,
        heading_jitter_rad=.35,minimum_pair_error_m=.12,entries=[dict(states=[asdict(s) for s in initial])])
    template_distances=formation_metrics(initial,1.,.25)['pair_distances_m']
    varied=False
    for seed in range(30):
        states=sample_initial_states(task,seed)
        assert states==sample_initial_states(task,seed)
        geometry=formation_metrics(states,1.,.25)
        assert geometry['pair_distances_m'].min()>=.7
        assert geometry['max_pair_error_m']>=.12
        assert not geometry['collision']
        assert all(-1.5<=s.x<=1.5 and -1.5<=s.y<=1.5 for s in states)
        assert all(math.isfinite(s.heading) and -math.pi<=s.heading<math.pi for s in states)
        varied |= not np.allclose(geometry['pair_distances_m'],template_distances)
    assert varied
    disabled=deepcopy(original);disabled['initialization']['hard_examples']=dict(probability=0.)
    assert sample_initial_states(disabled,54321)==sample_initial_states(original,54321)
    for bad in (-.1,1.1,float('nan')):
        invalid=deepcopy(task);invalid['initialization']['hard_examples']['probability']=bad
        try:sample_initial_states(invalid,1)
        except ValueError:pass
        else:raise AssertionError('Invalid probability accepted.')
    print('Seeded hard resets: physical clearance, variation and bounds: PASSED')
    print('Disabled hard sampling preserves legacy reset distribution: PASSED')

if __name__=='__main__':main()
