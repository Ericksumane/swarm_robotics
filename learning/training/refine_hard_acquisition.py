"""Fresh on-policy hard-start mining and a bounded safety/precision follow-up."""
from pathlib import Path
import copy,json,math,subprocess,sys
import numpy as np
import torch
from swarm_core.formation_env import FormationEnv
from swarm_core.learning.mappo_adapter import make_policy,hidden_inputs,pack

def main():
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[2];out=root/'experiments/runs/dispersed_hard_recovery_001';out.mkdir(exist_ok=False)
 selection=json.loads((root/'experiments/runs/dispersed_recovery_validation_001/selection.json').read_text());source=Path(selection['acquisition']['checkpoint']);state=torch.load(source,map_location='cpu',weights_only=True)
 task=json.load(open('configs/tasks/formation_relational_broad_evaluation.json'));comm=json.load(open('configs/communication/nominal.json'));reward=json.load(open('configs/training/dispersed_acquisition_reward.json'));env=FormationEnv(task,comm,reward)
 args,policy,_=make_policy(3,local_dim=9,critic_dim=state['critic_dim'],neighbor_dim=9,geometry_summary=True);policy.actor.load_state_dict(state['actor']);policy.actor.eval();h,m=hidden_inputs(3,args);entries=[];mining=[]
 for seed in range(76000000,76000040):
  obs,_=env.reset(seed);initial=[dict(x=s.x,y=s.y,heading=s.heading) for s in env.simulation.states]
  while True:
   with torch.no_grad():actions,_=policy.act(pack(obs),h,m,deterministic=True)
   obs,_,term,trunc,info=env.step(actions.numpy())
   if term or trunc:break
  mining.append(dict(seed=seed,reason=info['reason']))
  if info['reason']!='success':entries.append(dict(seed=seed,reason=info['reason'],states=initial))
 # Constructed initial poses only, not demonstrated controls or rollout edits.
 entries.append(dict(source='close head-on pair',states=[dict(x=-.36,y=0.,heading=0.),dict(x=.36,y=0.,heading=math.pi),dict(x=0.,y=1.5,heading=-math.pi/2)]))
 entries.append(dict(source='close third robot behind pair',states=[dict(x=-.36,y=0.,heading=0.),dict(x=.36,y=0.,heading=math.pi),dict(x=0.,y=-1.5,heading=math.pi/2)]))
 training=json.load(open('configs/tasks/dispersed_acquisition_training.json'));training['task_name']='hard_disperse_precision_training_v2';training['mission']['formation_tolerance_m']=.05
 training['initialization']['hard_examples']=dict(probability=.4,entries=entries,position_jitter_m=.02,heading_jitter_rad=.2,minimum_pair_error_m=.12)
 training_path=out/'training_task.json';training_path.write_text(json.dumps(training,indent=2));(out/'mining.json').write_text(json.dumps(dict(seed=76000000,cases=mining,bank_entries=len(entries)),indent=2))
 reward.update(reward_version='hard_disperse_precision_safety_v2',formation_weight=10.,proximity_weight=50.,proximity_margin_m=.4,collision_penalty=400.)
 reward_path=out/'training_reward.json';reward_path.write_text(json.dumps(reward,indent=2))
 with (out/'training.log').open('w') as f:subprocess.run([sys.executable,'scripts/train_formation.py','--steps','30000','--snapshot-interval','15000','--seed','457','--actor-lr','.00005','--gamma','.999','--gae-lambda','.99','--initial-std','.04','--final-std','.015','--std-decay-steps','20000','--entropy-coef','0','--resume',str(source),'--task',str(training_path),'--reward',str(reward_path),'--output',str(out/'training')],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Hard acquisition training completed.',flush=True)

if __name__=='__main__':main()
