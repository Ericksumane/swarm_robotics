"""Measure whether exploration alone disrupts an already correct formation."""
import json,math
from pathlib import Path
import numpy as np
from swarm_core.formation_env import FormationEnv
from swarm_core.dynamics import RobotState

def main():
 task=json.loads(Path('configs/tasks/formation_relative_critic.json').read_text())
 comm=json.loads(Path('configs/communication/nominal.json').read_text())
 weights=json.loads(Path('configs/training/formation_reward_proximity.json').read_text())
 triangle=[RobotState(x=0,y=0),RobotState(x=1,y=0),RobotState(x=.5,y=math.sqrt(3)/2)]
 results=[]
 for std in [0.,.02,.05,.1,.3]:
  success=0;max_hold=[]
  for seed in range(30):
   env=FormationEnv(task,comm,weights);env.reset(seed)
   env.simulation.states=triangle;env.simulation._measure();env.simulation._transmit();env.simulation.network.deliver(0.)
   rng=np.random.default_rng(123000+seed);hold=0.
   while True:
    _,_,term,trunc,info=env.step(rng.normal(0,std,(3,2)))
    hold=max(hold,info['hold_time_s'])
    if term or trunc:
     success+=info['reason']=='success';max_hold.append(hold);break
  row=dict(action_std=std,successes=success,episodes=30,mean_max_hold_s=float(np.mean(max_hold)))
  print(json.dumps(row),flush=True);results.append(row)
 Path('experiments/runs/stopping_exploration_audit.json').write_text(json.dumps(results,indent=2))
 print('Controlled zero-mean action noise only; not learned policy performance.')

if __name__=='__main__':main()
