"""Check seeded near-formation placement and physical separation."""
import json
import numpy as np
from swarm_core.initialization import sample_initial_states
from swarm_core.geometry import formation_metrics
def main():
 task=json.load(open('configs/tasks/formation_curriculum.json'))
 task['initialization']['near_formation']['probability']=1.
 for seed in range(100):
  states=sample_initial_states(task,seed)
  assert states==sample_initial_states(task,seed)
  metrics=formation_metrics(states,1.,.25)
  assert not metrics['collision']
  assert metrics['pair_distances_m'].min()>=task['initialization']['min_center_distance_m']
  assert metrics['max_pair_error_m']<=2*np.sqrt(2)*.12+1e-9
 print('Seeded near-formation curriculum and separation: PASSED')
if __name__=='__main__':main()
