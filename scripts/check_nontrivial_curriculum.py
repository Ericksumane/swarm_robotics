"""The acquisition curriculum must require action instead of already succeeding."""
import json
from swarm_core.initialization import sample_initial_states
from swarm_core.geometry import formation_metrics
def main():
 task=json.load(open('configs/tasks/formation_nontrivial_curriculum.json'))
 task['initialization']['near_formation']['probability']=1.
 for seed in range(100):
  states=sample_initial_states(task,seed)
  metrics=formation_metrics(states,1.,.25)
  assert not metrics['collision']
  assert metrics['max_pair_error_m']>=.12
  assert metrics['max_pair_error_m']>task['mission']['formation_tolerance_m']
  assert states==sample_initial_states(task,seed)
 print('Nontrivial formation acquisition starts across 100 seeds: PASSED')
if __name__=='__main__':main()
